"""
扶梯振动阈值自动确定 — 一体化脚本（Athena 直连）
==================================================
从 AWS Athena 直接查询数据，不生成中间 CSV 文件
数据直接传入算法（Otsu / GMM / KDE / K-means）

用法：修改下方设备号和日期范围，运行 python suanfa.py

依赖：numpy, pandas, scikit-learn, pyathena, boto3
安装：pip install numpy pandas scikit-learn pyathena boto3

设计：电流数据为可选参考，阈值仅从振动数据分布求得
"""

import numpy as np
import pandas as pd
import warnings

# 可选依赖：scikit-learn（GMM、KDE）
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import KernelDensity
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# 屏蔽 pyathena 直连触发的 SQLAlchemy 警告（无害，不影响查询）
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")


# ============================================================
# Athena 数据库连接
# ============================================================

class Athena_Connector:
    def __init__(self):
        import pyathena
        import boto3
        self.conn = pyathena.connect(
            s3_staging_dir="s3://athena-query-prod-zhen/",
            session=boto3.Session(profile_name='AWSPowerUserAccess-579289528406')
        )

    def query(self, sql):
        return pd.read_sql(sql, self.conn)

    def close(self):
        self.conn.close()


# ============================================================
# 算法 1：大津算法 Otsu（无监督，纯 numpy）
# ============================================================

def otsu_threshold(data: np.ndarray, bins: int = 256) -> float:
    """大津算法：自动计算使类间方差最大化的阈值"""
    data = data[np.isfinite(data)]
    if len(data) == 0:
        raise ValueError("数据为空或全部为 NaN")

    data_min, data_max = data.min(), data.max()
    if data_max == data_min:
        return float(data_min)

    hist, bin_edges = np.histogram(data, bins=bins)
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = hist.sum()

    w0_cumsum = np.cumsum(hist)
    w1_cumsum = total - w0_cumsum

    weighted = bin_centers * hist
    sum0_cumsum = np.cumsum(weighted)
    sum_total = sum0_cumsum[-1]
    sum1_cumsum = sum_total - sum0_cumsum

    valid = (w0_cumsum > 0) & (w1_cumsum > 0)

    mu0 = np.zeros_like(w0_cumsum)
    mu1 = np.zeros_like(w1_cumsum)
    mu0[valid] = sum0_cumsum[valid] / w0_cumsum[valid]
    mu1[valid] = sum1_cumsum[valid] / w1_cumsum[valid]

    var_between = np.zeros_like(w0_cumsum)
    var_between[valid] = (
        w0_cumsum[valid] * w1_cumsum[valid]
        * (mu0[valid] - mu1[valid]) ** 2
    )

    best_idx = np.argmax(var_between)
    return float(bin_centers[best_idx])


# ============================================================
# 算法 2：高斯混合模型 GMM（无监督，需 scikit-learn）
# ============================================================

def gmm_threshold(data: np.ndarray) -> float:
    """GMM：拟合两个高斯分布，取交点作为阈值"""
    if not _HAS_SKLEARN:
        raise ImportError("gmm_threshold 需要 scikit-learn: pip install scikit-learn")

    data = data[np.isfinite(data)].reshape(-1, 1)
    if len(data) < 10:
        raise ValueError("数据太少，无法拟合 GMM")

    gmm = GaussianMixture(n_components=2, random_state=42, n_init=3)
    gmm.fit(data)

    mu = gmm.means_.flatten()
    sigma = np.sqrt(gmm.covariances_.flatten())
    w = gmm.weights_

    if mu[0] > mu[1]:
        mu = mu[::-1]; sigma = sigma[::-1]; w = w[::-1]

    a = 1.0 / (2 * sigma[0]**2) - 1.0 / (2 * sigma[1]**2)
    b = mu[1] / sigma[1]**2 - mu[0] / sigma[0]**2
    c = (mu[0]**2 / (2 * sigma[0]**2) - mu[1]**2 / (2 * sigma[1]**2)
         + np.log(w[1] / w[0]) + np.log(sigma[0] / sigma[1]))

    if abs(a) < 1e-12:
        t = -c / b
    else:
        disc = b**2 - 4 * a * c
        if disc < 0:
            t = (mu[0] + mu[1]) / 2
        else:
            r1 = (-b + np.sqrt(disc)) / (2 * a)
            r2 = (-b - np.sqrt(disc)) / (2 * a)
            candidates = [r for r in (r1, r2) if mu[0] <= r <= mu[1]]
            t = candidates[0] if candidates else (mu[0] + mu[1]) / 2
    return float(t)


# ============================================================
# 算法 3：核密度估计 KDE（无监督，需 scikit-learn）
# ============================================================

def kde_threshold(data: np.ndarray, bandwidth: str = "scott") -> float:
    """KDE：估计概率密度曲线，找两峰之间的密度谷底作为阈值"""
    if not _HAS_SKLEARN:
        raise ImportError("kde_threshold 需要 scikit-learn: pip install scikit-learn")

    data = data[np.isfinite(data)]
    if len(data) < 10:
        raise ValueError("数据太少，无法做 KDE")

    x = data.reshape(-1, 1)
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(x)

    grid = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
    log_density = kde.score_samples(grid)
    density = np.exp(log_density)
    grid_x = grid[:, 0]

    # 找所有局部极大值（峰）
    peaks = []
    for i in range(1, len(density) - 1):
        if density[i] > density[i - 1] and density[i] >= density[i + 1]:
            peaks.append((grid_x[i], density[i]))

    if len(peaks) < 2:
        return otsu_threshold(data)

    # 取密度最高的两个峰
    peaks.sort(key=lambda p: p[1], reverse=True)
    peak1_x, peak2_x = peaks[0][0], peaks[1][0]
    lo, hi = min(peak1_x, peak2_x), max(peak1_x, peak2_x)

    # 只在两个峰之间找谷底
    valleys = []
    for i in range(1, len(density) - 1):
        if density[i] < density[i - 1] and density[i] < density[i + 1]:
            if lo <= grid_x[i] <= hi:
                valleys.append((grid_x[i], density[i]))

    if not valleys:
        return otsu_threshold(data)

    valleys.sort(key=lambda v: v[1])
    return float(valleys[0][0])


# ============================================================
# 算法 4：K-means 聚类中点（无监督）
# ============================================================

def kmeans_threshold(data: np.ndarray) -> float:
    """K-means (k=2)：聚成两类，阈值取两簇中心中点"""
    data = data[np.isfinite(data)].reshape(-1, 1)
    if len(data) < 4:
        raise ValueError("数据太少，无法聚类")

    if _HAS_SKLEARN:
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        km.fit(data)
        centers = np.sort(km.cluster_centers_.flatten())
    else:
        # 纯 numpy 实现
        c0, c1 = np.percentile(data, [25, 75])
        for _ in range(50):
            d0 = np.abs(data - c0)
            d1 = np.abs(data - c1)
            mask0 = d0 < d1
            if mask0.sum() == 0 or mask0.sum() == len(data):
                break
            c0 = data[mask0].mean()
            c1 = data[~mask0].mean()
        centers = np.sort([c0, c1])

    return float((centers[0] + centers[1]) / 2)


# ============================================================
# 纯振动数据清洗（无需电流）
# ============================================================

def clean_vibration_standalone(
    vibration: np.ndarray,
    z_score_cutoff: float = 6.0,
    percentile_clip: float = 0.1,
) -> tuple:
    """
    纯振动数据清洗：不依赖电流，仅从振动数据本身检测异常

    策略（三重过滤）:
      1. 去除 NaN / Inf
      2. 百分位截断：去掉最高和最低 percentile_clip% 的极端值
      3. 极宽松 MAD 检测：z-score > 6 才删（只删极端离群点）

    返回:
        cleaned_vibration : 清洗后的振动数组
        is_valid          : 布尔掩码（与原数组等长）
    """
    vibration = np.asarray(vibration, dtype=np.float64)
    n = len(vibration)
    is_valid = np.ones(n, dtype=bool)

    is_valid &= np.isfinite(vibration)
    if is_valid.sum() < 10:
        return vibration[is_valid], is_valid

    v = vibration[is_valid]

    p_low = np.percentile(v, percentile_clip)
    p_high = np.percentile(v, 100 - percentile_clip)
    v_clipped = v[(v >= p_low) & (v <= p_high)]

    if len(v_clipped) < 5:
        return vibration[is_valid], is_valid

    med = np.median(v_clipped)
    mad = np.median(np.abs(v_clipped - med))

    if mad > 0:
        valid_indices = np.where(is_valid)[0]
        z_scores = 0.6745 * (vibration[valid_indices] - med) / mad
        outliers = np.abs(z_scores) > z_score_cutoff
        is_valid[valid_indices[outliers]] = False

    cleaned = vibration[is_valid]
    return cleaned, is_valid


# ============================================================
# 双峰安全裕度分析（无需电流）
# ============================================================

def analyze_bimodal_margin(
    data: np.ndarray,
    threshold: float,
    n_grid: int = 1000,
) -> dict:
    """
    无需电流，纯从振动分布分析阈值的安全裕度

    用阈值把数据分两群，计算停机群 P99 和运行群 P1，
    衡量阈值到两者的距离（安全裕度）
    """
    data = data[np.isfinite(data)]
    if len(data) < 20:
        return {"error": "数据太少"}

    lower = data[data <= threshold]
    upper = data[data > threshold]

    if len(lower) < 5 or len(upper) < 5:
        return {"error": "阈值切分后某类样本太少"}

    stop_p99 = np.percentile(lower, 99)
    run_p1 = np.percentile(upper, 1)

    margin_to_stop = threshold - stop_p99
    margin_to_run = run_p1 - threshold
    min_margin = min(margin_to_stop, margin_to_run)
    gap_width = run_p1 - stop_p99

    hist, edges = np.histogram(data, bins=min(n_grid, 200))
    centers = (edges[:-1] + edges[1:]) / 2

    lower_mask = centers <= threshold
    upper_mask = centers > threshold

    if lower_mask.sum() > 0:
        stop_peak_idx = np.argmax(hist[lower_mask])
        stop_peak = centers[lower_mask][stop_peak_idx]
    else:
        stop_peak = np.median(lower)

    if upper_mask.sum() > 0:
        run_peak_idx = np.argmax(hist[upper_mask])
        run_peak = centers[upper_mask][run_peak_idx]
    else:
        run_peak = np.median(upper)

    return {
        "stop_peak": float(stop_peak),
        "run_peak": float(run_peak),
        "stop_p99": float(stop_p99),
        "run_p1": float(run_p1),
        "margin_to_stop": float(margin_to_stop),
        "margin_to_run": float(margin_to_run),
        "min_margin": float(min_margin),
        "gap_width": float(gap_width),
    }


# ============================================================
# 多算法横向对比（纯振动，无需电流）
# ============================================================

def compare_thresholds_standalone(
    vibration: np.ndarray,
    bins: int = 256,
    verbose: bool = True,
) -> dict:
    """纯振动数据多算法对比：不依赖电流"""
    vibration = np.asarray(vibration, dtype=np.float64)

    cleaned, is_valid = clean_vibration_standalone(vibration)
    removed = len(vibration) - is_valid.sum()

    if verbose:
        print(f"  [清洗] 移除 {removed} 个异常样本，保留 {is_valid.sum()} 个")

    results = {}

    algorithms = [
        ("Otsu", lambda d: otsu_threshold(d, bins=bins)),
        ("K-means", lambda d: kmeans_threshold(d)),
        ("GMM", lambda d: gmm_threshold(d) if _HAS_SKLEARN else None),
        ("KDE", lambda d: kde_threshold(d) if _HAS_SKLEARN else None),
    ]

    for name, algo_fn in algorithms:
        try:
            t = algo_fn(cleaned)
            if t is None:
                results[name] = {"threshold": None, "error": "依赖未安装"}
                continue
            margin = analyze_bimodal_margin(cleaned, t)
            results[name] = {
                "threshold": t,
                "margin_info": margin,
            }
        except Exception as e:
            results[name] = {"threshold": None, "error": str(e)}

    if verbose:
        print(f"\n  {'算法':<12} {'阈值':>8} {'停机P99':>8} {'运行P1':>8} "
              f"{'到停裕度':>8} {'到运裕度':>8} {'最小裕度':>8} {'无人区宽':>8}")
        print("  " + "-" * 78)
        for name, r in results.items():
            if r["threshold"] is None:
                print(f"  {name:<12} {'N/A':>8}  ({r.get('error', '')})")
                continue
            m = r["margin_info"]
            if "error" in m:
                print(f"  {name:<12} {r['threshold']:>8.3f}  (裕度分析失败)")
                continue
            print(f"  {name:<12} {r['threshold']:>8.3f} {m['stop_p99']:>8.3f} "
                  f"{m['run_p1']:>8.3f} {m['margin_to_stop']:>+8.3f} "
                  f"{m['margin_to_run']:>+8.3f} {m['min_margin']:>+8.3f} "
                  f"{m['gap_width']:>8.3f}")

    return results


# ============================================================
# 一站式无监督分析（纯振动，无需电流）
# ============================================================

def analyze_vibration_standalone(
    vibration: np.ndarray,
    bins: int = 256,
    verbose: bool = True,
) -> dict:
    """
    一站式纯振动分析：清洗 → 多算法求阈值 → 裕度分析 → 推荐

    返回:
        dict: {
            "cleaned_count", "removed_count",
            "all_thresholds",   — 各算法阈值
            "margins",          — 各算法裕度
            "best_algorithm",   — 最优算法
            "best_threshold",   — 推荐阈值
            "is_valid",         — 清洗掩码
        }
    """
    vibration = np.asarray(vibration, dtype=np.float64)

    cleaned, is_valid = clean_vibration_standalone(vibration)
    removed = len(vibration) - is_valid.sum()

    if verbose:
        print(f"  [清洗] 移除 {removed} 个异常样本，保留 {is_valid.sum()} 个")

    results = compare_thresholds_standalone(vibration, bins=bins, verbose=verbose)

    best_algo = None
    best_margin = -1e9
    for name, r in results.items():
        if r["threshold"] is None:
            continue
        m = r.get("margin_info", {})
        if "min_margin" not in m:
            continue
        if m["min_margin"] > best_margin:
            best_margin = m["min_margin"]
            best_algo = name

    best_t = results[best_algo]["threshold"] if best_algo else None

    if verbose and best_algo:
        print(f"\n  [推荐] 算法={best_algo}, 阈值={best_t:.4f}, "
              f"最小裕度={best_margin:.3f}")

    return {
        "cleaned_count": int(is_valid.sum()),
        "removed_count": removed,
        "all_thresholds": {n: r["threshold"] for n, r in results.items()
                           if r["threshold"] is not None},
        "margins": {n: r.get("margin_info", {})
                    for n, r in results.items() if r["threshold"] is not None},
        "best_algorithm": best_algo,
        "best_threshold": best_t,
        "is_valid": is_valid,
    }


# ============================================================
# 阈值评估（有电流时用，无电流时跳过）
# ============================================================

def evaluate_threshold(
    vibration: np.ndarray,
    current: np.ndarray,
    current_threshold: float,
    vib_threshold: float,
) -> dict:
    """以电流判断为准真值，评估振动阈值的准确率/覆盖率"""
    true_running = current > current_threshold
    pred_running = vibration > vib_threshold

    tp = np.sum(pred_running & true_running)
    fp = np.sum(pred_running & ~true_running)
    fn = np.sum(~pred_running & true_running)
    tn = np.sum(~pred_running & ~true_running)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
    }


# ============================================================
# 电流分界点自动检测（各算法）
# ============================================================

def find_current_threshold(
    current: np.ndarray,
    method: str = "auto",
    bins: int = 256,
    verbose: bool = True,
    equipment_ids: np.ndarray = None,
) -> float:
    """
    用多种算法自动检测电流数据的分界点（停机 vs 运行）

    电流数据本身也是双峰分布（停机 ~0A，运行 ~12A），
    所以用与振动相同的算法来找分界点。

    如果有多台设备且电流范围差异大，会按设备分别检测，
    取各设备分界点的中位数作为全局分界点。

    参数:
        current : 电流数据（三相均值或其他汇总值）
        method  : 算法选择
            "auto"    — 自动选最优（选裕度最大的）
            "otsu"    — 大津算法
            "kmeans"  — K-means 聚类中点
            "gmm"     — 高斯混合模型交点
            "kde"     — 核密度估计谷底
        bins       : Otsu 分桶数
        verbose    : 打印过程
        equipment_ids : 设备编号数组（与 current 等长），按设备分别检测

    返回:
        current_threshold : 电流分界点
    """
    current = np.asarray(current, dtype=np.float64)
    cur_clean = current[np.isfinite(current)]

    if len(cur_clean) < 10:
        raise ValueError("电流数据太少")

    cmin, cmax = cur_clean.min(), cur_clean.max()
    if cmax - cmin < 0.01:
        raise ValueError("电流数据几乎全相同，无法检测分界点")

    # 如果有设备编号，检查各设备电流范围是否差异大
    if equipment_ids is not None:
        eq_arr = np.asarray(equipment_ids)
        unique_eqs = np.unique(eq_arr[np.isfinite(current)])
        if len(unique_eqs) > 1:
            eq_ranges = {}
            for eq in unique_eqs:
                eq_cur = current[eq_arr == eq]
                eq_cur = eq_cur[np.isfinite(eq_cur)]
                if len(eq_cur) > 0:
                    eq_ranges[str(eq)] = (eq_cur.min(), eq_cur.max())

            max_values = [r[1] for r in eq_ranges.values()]
            if max(max_values) / (min(max_values) + 1e-9) > 2:
                if verbose:
                    print(f"  检测到 {len(unique_eqs)} 台设备，电流范围差异大，"
                          f"按设备分别检测（每台一行摘要）:")

                eq_thresholds = []
                for eq in unique_eqs:
                    eq_mask = eq_arr == eq
                    eq_cur = current[eq_mask]
                    eq_cur = eq_cur[np.isfinite(eq_cur)]
                    if len(eq_cur) < 10 or eq_cur.max() - eq_cur.min() < 0.01:
                        if verbose:
                            print(f"    设备 {eq}: 电流范围太小，跳过")
                        continue
                    # 电流数据不做 MAD 清洗（双峰差距大，MAD 会误删停机段）
                    # 只去掉 NaN/Inf 即可。verbose=False 关闭每台设备的详细输出
                    t = _detect_single_current(
                        eq_cur, method, bins, False, str(eq)
                    )
                    if t is not None:
                        eq_thresholds.append(t)
                        if verbose:
                            n_r = np.sum(eq_cur > t)
                            n_s = np.sum(eq_cur <= t)
                            print(f"    设备 {eq}: 范围[{eq_cur.min():.2f}, "
                                  f"{eq_cur.max():.2f}]  "
                                  f"分界点={t:.4f}  "
                                  f"运行{n_r}条/停机{n_s}条")

                if len(eq_thresholds) == 0:
                    raise ValueError("所有设备的电流都无法检测分界点")

                best_t = float(np.median(eq_thresholds))
                if verbose:
                    print(f"\n  各设备分界点: {[f'{t:.4f}' for t in eq_thresholds]}")
                    print(f"  -> 全局电流分界点(中位数): {best_t:.4f}")

                n_run = np.sum(cur_clean > best_t)
                n_stop = np.sum(cur_clean <= best_t)
                if n_run == 0 or n_stop == 0:
                    if verbose:
                        print(f"  警告: 全局分界点 {best_t:.4f} 导致某类为 0")
                return best_t

    # 电流数据不做 MAD 清洗（双峰差距大，MAD 会误删停机段）
    cur_cleaned = cur_clean
    best_t = _detect_single_current(cur_cleaned, method, bins, verbose, None)
    if best_t is None:
        raise ValueError("无法检测电流分界点")
    return float(best_t)


def _detect_single_current(cur_cleaned, method, bins, verbose, label=None):
    """对单个电流数据集检测分界点（内部辅助函数）"""
    results = {}
    results["Otsu"] = otsu_threshold(cur_cleaned, bins=bins)
    try:
        results["K-means"] = kmeans_threshold(cur_cleaned)
    except:
        pass
    if _HAS_SKLEARN:
        try:
            results["GMM"] = gmm_threshold(cur_cleaned)
        except:
            pass
        try:
            results["KDE"] = kde_threshold(cur_cleaned)
        except:
            pass

    cmin, cmax = cur_cleaned.min(), cur_cleaned.max()
    prefix = f"  设备 {label} " if label else "  "

    if verbose:
        print(f"{prefix}电流范围: [{cmin:.3f}, {cmax:.3f}]")
        print(f"{prefix}各算法分界点:")
        print(f"{prefix}{'算法':<10} {'分界点':>8} {'停机%':>7} {'运行%':>7} "
              f"{'停机P99':>8} {'运行P1':>8} {'裕度':>8}")
        print(prefix + "-" * 70)

    best_t = None
    best_score = -1e9
    best_name = None

    for name, t in results.items():
        margin = analyze_bimodal_margin(cur_cleaned, t)
        if "error" in margin:
            if verbose:
                print(f"{prefix}{name:<10} {t:>8.3f}  (失败)")
            continue

        n_run = np.sum(cur_cleaned > t)
        n_stop = np.sum(cur_cleaned <= t)
        pct_run = n_run / len(cur_cleaned) * 100
        pct_stop = n_stop / len(cur_cleaned) * 100

        if verbose:
            print(f"{prefix}{name:<10} {t:>8.3f} {pct_stop:>6.1f}% {pct_run:>6.1f}% "
                  f"{margin['stop_p99']:>8.3f} {margin['run_p1']:>8.3f} "
                  f"{margin['min_margin']:>+8.3f}")

        balance = 1.0 - abs(pct_run - 50) / 50
        margin_norm = margin['min_margin'] / (cmax - cmin + 1e-9)
        score = margin_norm * 0.7 + balance * 0.3

        if score > best_score:
            best_score = score
            best_t = t
            best_name = name

    if method != "auto":
        method_map = {"otsu": "Otsu", "kmeans": "K-means",
                      "gmm": "GMM", "kde": "KDE"}
        key = method_map.get(method.lower(), method)
        if key in results:
            best_t = results[key]
            best_name = key

    if verbose and best_name:
        tag = "指定" if method != "auto" else "推荐"
        print(f"{prefix}-> {tag}算法 {best_name}, 分界点 = {best_t:.4f}")

    if best_t is not None:
        n_run = np.sum(cur_cleaned > best_t)
        n_stop = np.sum(cur_cleaned <= best_t)
        if n_run == 0 or n_stop == 0:
            if verbose:
                print(f"{prefix}警告: 分界点 {best_t:.4f} 导致某类为 0")

    return best_t


# ============================================================
# ============================================================
#  以下为主程序
# ============================================================
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # ============================================================
    # ★★★ 修改这里：改成你要查询的设备和日期范围 ★★★
    # ============================================================
    EQUIPMENT_NUMBERS = ['30248586']          # 设备号列表，可多台
    START_DATE = '2026-07-10'                 # 开始日期
    END_DATE = '2026-07-15'                   # 结束日期

    # 振动列名
    VIBRATION_COL = 'gbvibforwardrms'
    # 电流列名（如果没有电流数据，设为 None）
    CURRENT_COLS = ['motorcurrent1avg', 'motorcurrent2avg', 'motorcurrent3avg']
    # 电流阈值：设为 None 表示自动检测（推荐），也可以手动指定如 2.0
    CURRENT_THRESHOLD = None
    # 电流分界点检测算法（仅当 CURRENT_THRESHOLD = None 时生效）:
    #   "auto"   — 自动选最优（裕度最大的算法）
    #   "otsu"   — 大津算法
    #   "kmeans" — K-means 聚类中点
    #   "gmm"    — 高斯混合模型交点（需 scikit-learn）
    #   "kde"    — 核密度估计谷底（需 scikit-learn）
    CURRENT_METHOD = "auto"

    # ★★★ 训练/测试分界日期 ★★★
    # <= SPLIT_DATE 的数据用来求阈值（训练集）
    # >  SPLIT_DATE 的数据用来验证阈值（测试集）
    # 设为 None 表示不分训练/测试，全部数据一起求阈值+验证
    SPLIT_DATE = None                          # 如 '2026-07-11'

    # ============================================================
    # 1. 从 Athena 查询数据（不生成 CSV，直接用）
    # ============================================================
    print("=" * 78)
    print("扶梯振动阈值自动确定 — 多算法 + Athena 直连")
    print("=" * 78)

    print(f"\n[数据源] AWS Athena 直连")
    print(f"  设备: {EQUIPMENT_NUMBERS}")
    print(f"  日期范围: {START_DATE} ~ {END_DATE}")

    eq_list = ", ".join(f"'{e}'" for e in EQUIPMENT_NUMBERS)
    sql = f"""
        SELECT equipmentnumber, gbvibforwardrms, motorcurrent1avg, motorcurrent2avg,
               motorcurrent3avg, modeset, operationstatus,
               (FROM_ISO8601_TIMESTAMP(timestamp) AT TIME ZONE 'Asia/Shanghai') AS event_time_shanghai
        FROM data_cleansed."anyescalator"
        WHERE equipmentnumber IN ({eq_list})
          AND eventdate BETWEEN '{START_DATE}' AND '{END_DATE}'
        ORDER BY timestamp
    """

    conn = Athena_Connector()
    try:
        combined = conn.query(sql)
    finally:
        conn.close()

    print(f"  查询到 {len(combined)} 条数据")
    print(f"  列名: {list(combined.columns)}")

    if len(combined) == 0:
        print("\n[错误] 没有查询到数据！请检查设备号和日期范围。")
        exit(1)

    # 去重（如果有时间和设备号列）
    dedup_cols = [c for c in ['equipmentnumber', 'event_time_shanghai']
                  if c in combined.columns]
    if len(dedup_cols) == 2:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
        if before != len(combined):
            print(f"\n去重: {before} -> {len(combined)} 条")

    print(f"\n合并后总行数: {len(combined)}")

    # ============================================================
    # 2. 训练/测试集划分（按日期切分）
    # ============================================================
    use_split = (SPLIT_DATE is not None
                 and 'event_time_shanghai' in combined.columns)

    train_df = combined
    test_df = None

    if use_split:
        # 用字符串前10位（YYYY-MM-DD）比较，避免时区问题
        combined['date_str'] = combined['event_time_shanghai'].astype(str).str[:10]
        train_df = combined[combined['date_str'] <= SPLIT_DATE].copy()
        test_df = combined[combined['date_str'] > SPLIT_DATE].copy()

        print(f"\n训练/测试集划分:")
        print(f"  训练集: {len(train_df)} 条（{train_df['date_str'].min()} "
              f"~ {train_df['date_str'].max()}）")
        print(f"  测试集: {len(test_df)} 条（{test_df['date_str'].min()} "
              f"~ {test_df['date_str'].max()}）")

        # 用 train_df 替代 combined 做后续阈值分析
        analysis_df = train_df
    else:
        analysis_df = combined

    # ============================================================
    # 3. 提取振动数据
    # ============================================================
    if VIBRATION_COL not in analysis_df.columns:
        print(f"\n[错误] 数据中没有找到振动列 '{VIBRATION_COL}'！")
        print(f"  列名: {list(analysis_df.columns)}")
        print(f"  请修改 VIBRATION_COL 为正确的列名")
        exit(1)

    vibration = analysis_df[VIBRATION_COL].values.astype(float)

    # 检查是否有电流数据
    has_current = all(c in analysis_df.columns for c in CURRENT_COLS) if CURRENT_COLS else False
    current_mean = None
    if has_current:
        current1 = analysis_df[CURRENT_COLS[0]].values.astype(float)
        current2 = analysis_df[CURRENT_COLS[1]].values.astype(float)
        current3 = analysis_df[CURRENT_COLS[2]].values.astype(float)
        current_mean = (current1 + current2 + current3) / 3.0
        valid_cur = current_mean[np.isfinite(current_mean)]

        print(f"\n电流数据: 可用")

        if len(valid_cur) < 10 or valid_cur.max() - valid_cur.min() < 0.01:
            print(f"  警告: 电流数据太少或几乎全相同，无法用于验证")
            has_current = False
            current_mean = None
        elif CURRENT_THRESHOLD is None:
            # 自动检测电流分界点（多算法对比）
            print(f"\n  [自动检测电流分界点 — 多算法对比]")
            try:
                CURRENT_THRESHOLD = find_current_threshold(
                    current_mean,
                    method=CURRENT_METHOD,
                    verbose=True,
                    equipment_ids=analysis_df['equipmentnumber'].values if 'equipmentnumber' in analysis_df.columns else None,
                )
                print(f"\n  -> 最终电流分界点: {CURRENT_THRESHOLD:.4f}")
            except Exception as e:
                print(f"  电流分界点检测失败: {e}")
                has_current = False
                current_mean = None
        else:
            # 手动指定
            n_run = np.sum(valid_cur > CURRENT_THRESHOLD)
            n_stop = np.sum(valid_cur <= CURRENT_THRESHOLD)
            print(f"  使用手动电流阈值: {CURRENT_THRESHOLD}")
            print(f"  电流 > 阈值（运行）: {n_run} 条")
            print(f"  电流 <= 阈值（停机）: {n_stop} 条")
            if n_run == 0:
                print(f"\n  警告: 没有电流 > {CURRENT_THRESHOLD} 的数据！")
                print(f"    电流最大值只有 {valid_cur.max():.3f}")
                print(f"    -> 建议改为 CURRENT_THRESHOLD = None 让脚本自动检测")
                has_current = False
                current_mean = None
    else:
        missing = [c for c in CURRENT_COLS if c not in analysis_df.columns] if CURRENT_COLS else []
        print(f"电流数据: 不可用（缺少 {missing}）")
        print(f"  -> 阈值仅从振动分布求得，不验证准确率")

    # 基本统计
    valid_vib = vibration[np.isfinite(vibration)]
    print(f"\n振动数据统计:")
    print(f"  有效样本: {len(valid_vib)}")
    print(f"  范围: [{valid_vib.min():.3f}, {valid_vib.max():.3f}]")
    print(f"  均值: {valid_vib.mean():.3f}, 中位数: {np.median(valid_vib):.3f}")

    if 'equipmentnumber' in analysis_df.columns:
        print(f"\n设备列表: {analysis_df['equipmentnumber'].unique()}")

    # ============================================================
    # 4. 全局纯振动分析（用训练集求阈值）
    # ============================================================
    print(f"\n{'='*78}")
    if use_split:
        print("全局纯振动分析（训练集求阈值）")
    else:
        print("全局纯振动分析（不依赖电流求阈值）")
    print(f"{'='*78}")

    result = analyze_vibration_standalone(vibration, bins=256, verbose=True)

    # ============================================================
    # 5. 有电流时验证（训练集内部验证）
    # ============================================================
    if has_current and current_mean is not None:
        print(f"\n{'='*78}")
        if use_split:
            print("训练集验证（电流可用，验证振动阈值的准确率/覆盖率）")
        else:
            print("附加验证（电流可用，验证振动阈值的准确率/覆盖率）")
        print(f"{'='*78}")

        best_t = result['best_threshold']
        is_valid = result['is_valid']
        valid_vib_arr = vibration[is_valid]
        valid_cur_arr = current_mean[is_valid]

        m = evaluate_threshold(valid_vib_arr, valid_cur_arr,
                                CURRENT_THRESHOLD, best_t)
        print(f"\n  推荐阈值 {best_t:.4f} 的验证结果:")
        print(f"    准确率 (Precision): {m['precision']:.4f}")
        print(f"    覆盖率 (Recall):    {m['recall']:.4f}")
        print(f"    F1-score:           {m['f1_score']:.4f}")
        print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

        print(f"\n  各算法阈值验证:")
        print(f"  {'算法':<12} {'阈值':>8} {'准确率':>8} {'覆盖率':>8} {'F1':>8}")
        print("  " + "-" * 50)
        for name, t in result['all_thresholds'].items():
            m = evaluate_threshold(valid_vib_arr, valid_cur_arr,
                                    CURRENT_THRESHOLD, t)
            print(f"  {name:<12} {t:>8.3f} {m['precision']:>8.4f} "
                  f"{m['recall']:>8.4f} {m['f1_score']:>8.4f}")

    # ============================================================
    # 5b. 测试集验证（用训练集求出的阈值验证新数据）
    # ============================================================
    if use_split and test_df is not None and len(test_df) >= 20:
        print(f"\n{'='*78}")
        print("验证 — 测试集（用训练集求出的阈值验证新数据）")
        print(f"{'='*78}")

        vibration_test = test_df[VIBRATION_COL].values.astype(float)
        cleaned_test, is_valid_test = clean_vibration_standalone(vibration_test)
        removed_test = len(vibration_test) - is_valid_test.sum()
        print(f"  [清洗] 测试集移除 {removed_test} 个异常样本，"
              f"保留 {is_valid_test.sum()} 个")

        vib_test_valid = vibration_test[is_valid_test]

        # 测试集裕度分析
        best_t = result['best_threshold']
        margin_test = analyze_bimodal_margin(cleaned_test, best_t)
        if "error" not in margin_test:
            print(f"\n  测试集裕度分析（阈值={best_t:.4f}）:")
            print(f"    停机 P99:     {margin_test['stop_p99']:.3f}")
            print(f"    运行 P1:      {margin_test['run_p1']:.3f}")
            print(f"    最小裕度:     {margin_test['min_margin']:+.3f}")
            print(f"    无人区宽度:   {margin_test['gap_width']:.3f}")

        # 如果测试集有电流数据，验证准确率/覆盖率
        test_has_current = (has_current and current_mean is not None
                           and all(c in test_df.columns for c in CURRENT_COLS))
        if test_has_current:
            test_cur1 = test_df[CURRENT_COLS[0]].values.astype(float)
            test_cur2 = test_df[CURRENT_COLS[1]].values.astype(float)
            test_cur3 = test_df[CURRENT_COLS[2]].values.astype(float)
            test_cur_mean = (test_cur1 + test_cur2 + test_cur3) / 3.0
            test_cur_valid = test_cur_mean[is_valid_test]

            print(f"\n  [电流验证] 测试集验证结果:")
            m_test = evaluate_threshold(vib_test_valid, test_cur_valid,
                                         CURRENT_THRESHOLD, best_t)
            print(f"    准确率 (Precision): {m_test['precision']:.4f}")
            print(f"    覆盖率 (Recall):    {m_test['recall']:.4f}")
            print(f"    F1-score:           {m_test['f1_score']:.4f}")
            print(f"    TP={m_test['tp']}  FP={m_test['fp']}  "
                  f"FN={m_test['fn']}  TN={m_test['tn']}")

            # 训练集 vs 测试集 对比
            if has_current and current_mean is not None:
                m_train = evaluate_threshold(
                    vibration[result['is_valid']],
                    current_mean[result['is_valid']],
                    CURRENT_THRESHOLD, best_t)

                print(f"\n  训练集 vs 测试集 对比:")
                print(f"    {'指标':<14} {'训练集':>10} {'测试集':>10} {'差值':>10}")
                print("    " + "-" * 44)
                for label, mk, mt in [
                    ("准确率", m_train['precision'], m_test['precision']),
                    ("覆盖率", m_train['recall'], m_test['recall']),
                    ("F1", m_train['f1_score'], m_test['f1_score']),
                ]:
                    diff = mt - mk
                    print(f"    {label:<14} {mk:>10.4f} {mt:>10.4f} {diff:>+10.4f}")

                if abs(m_test['f1_score'] - m_train['f1_score']) < 0.05:
                    print(f"\n  -> 训练集和测试集表现接近，阈值泛化能力良好")
                else:
                    print(f"\n  -> 训练集和测试集表现差异较大，"
                          f"可能需要重新评估阈值")
        else:
            print(f"\n  [电流验证] 测试集无电流数据，跳过准确率/覆盖率验证")

    # ============================================================
    # 6. 逐设备分析
    # ============================================================
    if 'equipmentnumber' in analysis_df.columns:
        print(f"\n{'='*78}")
        print("逐设备纯振动分析")
        print(f"{'='*78}")

        for equip in analysis_df['equipmentnumber'].unique():
            mask = analysis_df['equipmentnumber'] == equip
            v = vibration[mask]
            if len(v) < 20:
                continue

            print(f"\n  --- 设备 {equip} ({mask.sum()} 样本) ---")
            r = analyze_vibration_standalone(v, bins=256, verbose=True)
            if r['best_algorithm']:
                print(f"  推荐: {r['best_algorithm']}, "
                      f"阈值={r['best_threshold']:.4f}")

    # ============================================================
    # 7. 总结
    # ============================================================
    print(f"\n{'='*78}")
    print("总结")
    print(f"{'='*78}")
    if result['best_algorithm']:
        data_source_desc = (
            f"Athena 直连（设备 {EQUIPMENT_NUMBERS}，{START_DATE} ~ {END_DATE}）"
        )
        split_desc = (
            f"\n  训练/测试分界: {SPLIT_DATE}"
            if use_split else ""
        )
        print(f"""
  全局推荐算法: {result['best_algorithm']}
  全局推荐阈值: {result['best_threshold']:.4f}
  数据来源: {data_source_desc}{split_desc}
  清洗移除: {result['removed_count']} 样本
  有效样本: {result['cleaned_count']}

  各算法阈值:""")
        for name, t in result['all_thresholds'].items():
            margin = result['margins'].get(name, {})
            mm = margin.get('min_margin', 0)
            print(f"    {name:<12}: {t:.4f}  (最小裕度={mm:.3f})")

        print(f"""
  使用方法:
    当振动均值 >= {result['best_threshold']:.2f} 时，判定电梯在运行
    当振动均值 <  {result['best_threshold']:.2f} 时，判定电梯已停机

  注意:
    - 阈值仅从振动数据分布求得，不需要电流
    - 建议定期用新数据重新计算阈值（每月或每季度）
    - 不同设备振动范围可能不同，如需精确可按设备分别设阈值
""")
    else:
        print("\n  [警告] 未能确定推荐阈值，请检查数据质量")
