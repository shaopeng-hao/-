"""
扶梯振动阈值自动确定 — KDE算法 纯振动版（无需电流）
==================================================
只使用 KDE（核密度估计）算法求振动阈值
不需要电流数据，不需要 scikit-learn 以外的依赖

用法：把此文件放到 C:\\git 目录下，确保该目录下有 CSV 数据文件
运行：python kde_vibration_only.py

依赖：numpy, pandas, scikit-learn
安装：pip install numpy pandas scikit-learn
"""

import numpy as np
import pandas as pd
import glob
import os

from sklearn.neighbors import KernelDensity


# ============================================================
# KDE 核密度估计算法
# ============================================================

def kde_threshold(data: np.ndarray, bandwidth: str = "scott") -> float:
    """
    KDE：估计概率密度曲线，找两峰之间的密度谷底作为阈值

    原理：
      1. 用核函数（高斯核）对数据做平滑密度估计
      2. 在密度曲线上找所有局部极大值（峰）
      3. 取密度最高的两个峰（停机峰 + 运行峰）
      4. 在两峰之间找密度最低的谷底，即为阈值

    为什么 KDE 适合你的场景:
      - 停机时振动 ~0，运行时振动 ~4-8，两峰之间有无人区
      - KDE 找的谷底天然落在无人区正中央，安全裕度最大
      - 不假设数据服从高斯分布，对偏态/多峰都鲁棒

    参数:
        data      : 一维振动数据数组
        bandwidth : 核密度带宽，"scott" 为自动选择

    返回:
        threshold : 阈值（密度谷底对应的振动值）
    """
    data = data[np.isfinite(data)]
    if len(data) < 10:
        raise ValueError("数据太少，无法做 KDE")

    x = data.reshape(-1, 1)
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(x)

    # 生成密集网格点，计算每个点的密度
    grid = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
    log_density = kde.score_samples(grid)
    density = np.exp(log_density)
    grid_x = grid[:, 0]

    # 找所有局部极大值（峰）
    peaks = []
    for i in range(1, len(density) - 1):
        if density[i] > density[i - 1] and density[i] >= density[i + 1]:
            peaks.append((grid_x[i], density[i]))

    # 如果找不到 2 个峰，回退到大津算法
    if len(peaks) < 2:
        return _otsu_fallback(data)

    # 取密度最高的两个峰
    peaks.sort(key=lambda p: p[1], reverse=True)
    peak1_x, peak2_x = peaks[0][0], peaks[1][0]
    lo, hi = min(peak1_x, peak2_x), max(peak1_x, peak2_x)

    # 只在两个峰之间找谷底（密度最低点）
    valleys = []
    for i in range(1, len(density) - 1):
        if density[i] < density[i - 1] and density[i] < density[i + 1]:
            if lo <= grid_x[i] <= hi:
                valleys.append((grid_x[i], density[i]))

    if not valleys:
        return _otsu_fallback(data)

    # 取密度最低的谷底
    valleys.sort(key=lambda v: v[1])
    return float(valleys[0][0])


def _otsu_fallback(data: np.ndarray, bins: int = 256) -> float:
    """KDE 失败时的回退：大津算法"""
    data = data[np.isfinite(data)]
    hist, edges = np.histogram(data, bins=bins)
    total = len(data)
    best_var, best_t = 0, edges[0]
    for i in range(bins - 1):
        w0 = np.sum(hist[:i+1])
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        bc0 = (edges[:i+1] + edges[1:i+2]) / 2
        m0 = np.sum(hist[:i+1] * bc0) / w0
        bc1 = (edges[i+1:bins] + edges[i+2:bins+1]) / 2
        m1 = np.sum(hist[i+1:bins] * bc1) / w1
        var_between = w0 * w1 * (m0 - m1) ** 2 / (total ** 2)
        if var_between > best_var:
            best_var = var_between
            best_t = edges[i+1]
    return float(best_t)


# ============================================================
# 数据清洗
# ============================================================

def clean_vibration(vibration: np.ndarray,
                    z_score_cutoff: float = 6.0,
                    percentile_clip: float = 0.1) -> tuple:
    """
    振动数据清洗：去除异常值

    策略:
      1. 去除 NaN / Inf
      2. 百分位截断：去掉最高和最低 0.1% 的极端值
      3. MAD 检测：z-score > 6 才删（只删极端离群点）

    返回:
        cleaned : 清洗后的数组
        is_valid: 布尔掩码（与原数组等长）
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
# 双峰安全裕度分析
# ============================================================

def analyze_margin(data: np.ndarray, threshold: float) -> dict:
    """
    用阈值把数据分两群，计算安全裕度

    停机群 P99：停机数据中第 99 百分位的值（最大"接近运行"的停机值）
    运行群 P1 ：运行数据中第 1 百分位的值（最小"接近停机"的运行值）

    裕度 = min(阈值 - 停机P99, 运行P1 - 阈值)
    裕度越大，阈值越安全
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

    return {
        "stop_p99": float(stop_p99),
        "run_p1": float(run_p1),
        "margin_to_stop": float(threshold - stop_p99),
        "margin_to_run": float(run_p1 - threshold),
        "min_margin": float(min(threshold - stop_p99, run_p1 - threshold)),
        "gap_width": float(run_p1 - stop_p99),
    }


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # ★★★ 修改这里：改成你的 CSV 文件所在目录 ★★★
    # ============================================================
    DATA_DIR = r'C:\git\数据'

    # 振动列名（如果你的 CSV 列名不同，改这里）
    VIBRATION_COL = 'gbvibforwardrms'

    # ============================================================
    # 1. 加载所有 CSV 文件
    # ============================================================
    print("=" * 78)
    print("扶梯振动阈值自动确定 — KDE算法 纯振动版（无需电流）")
    print("=" * 78)

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    if len(csv_files) == 0:
        print(f"\n[错误] 在 {DATA_DIR} 目录下没有找到 CSV 文件！")
        exit(1)

    print(f"\n找到 {len(csv_files)} 个 CSV 文件:")
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig', nrows=5)
            cols = list(df.columns)
            print(f"  {os.path.basename(f)} (列: {cols[:6]}...)")
        except Exception as e:
            print(f"  {os.path.basename(f)} (读取失败: {e})")

    # 合并
    all_dfs = []
    for f in csv_files:
        try:
            all_dfs.append(pd.read_csv(f, encoding='utf-8-sig'))
        except Exception as e:
            print(f"  [跳过] {os.path.basename(f)}: {e}")

    if len(all_dfs) == 0:
        print("\n[错误] 没有成功加载任何 CSV 文件！")
        exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)

    # 去重
    dedup_cols = [c for c in ['equipmentnumber', 'event_time_shanghai']
                  if c in combined.columns]
    if len(dedup_cols) == 2:
        combined = combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    print(f"\n合并后总行数: {len(combined)}")

    # ============================================================
    # 2. 提取振动数据
    # ============================================================
    if VIBRATION_COL not in combined.columns:
        print(f"\n[错误] CSV 中没有找到振动列 '{VIBRATION_COL}'！")
        print(f"  CSV 中的列名: {list(combined.columns)}")
        exit(1)

    vibration = combined[VIBRATION_COL].values.astype(float)

    # 基本统计
    valid_vib = vibration[np.isfinite(vibration)]
    print(f"\n振动数据统计:")
    print(f"  有效样本: {len(valid_vib)}")
    print(f"  范围: [{valid_vib.min():.3f}, {valid_vib.max():.3f}]")
    print(f"  均值: {valid_vib.mean():.3f}, 中位数: {np.median(valid_vib):.3f}")

    if 'equipmentnumber' in combined.columns:
        print(f"  设备列表: {combined['equipmentnumber'].unique()}")

    # ============================================================
    # 3. KDE 求振动阈值
    # ============================================================
    print(f"\n{'='*78}")
    print("KDE 振动阈值分析")
    print(f"{'='*78}")

    cleaned, is_valid = clean_vibration(vibration)
    removed = len(vibration) - is_valid.sum()
    print(f"  [清洗] 移除 {removed} 个异常样本，保留 {is_valid.sum()} 个")

    threshold = kde_threshold(cleaned)
    margin = analyze_margin(cleaned, threshold)

    print(f"\n  KDE 阈值: {threshold:.4f}")
    if "error" in margin:
        print(f"  裕度分析: {margin['error']}")
    else:
        print(f"  停机 P99:     {margin['stop_p99']:.3f}")
        print(f"  运行 P1:      {margin['run_p1']:.3f}")
        print(f"  到停机裕度:   {margin['margin_to_stop']:+.3f}")
        print(f"  到运行裕度:   {margin['margin_to_run']:+.3f}")
        print(f"  最小裕度:     {margin['min_margin']:+.3f}")
        print(f"  无人区宽度:   {margin['gap_width']:.3f}")

    # ============================================================
    # 4. 逐设备分析
    # ============================================================
    if 'equipmentnumber' in combined.columns:
        print(f"\n{'='*78}")
        print("逐设备分析")
        print(f"{'='*78}")

        for equip in combined['equipmentnumber'].unique():
            mask = combined['equipmentnumber'] == equip
            v = vibration[mask]
            if len(v) < 20:
                continue

            v_clean, _ = clean_vibration(v)
            t = kde_threshold(v_clean)
            m = analyze_margin(v_clean, t)

            print(f"\n  设备 {equip} ({mask.sum()} 样本):")
            print(f"    KDE 阈值: {t:.4f}")
            if "error" not in m:
                print(f"    最小裕度: {m['min_margin']:+.3f}")

    # ============================================================
    # 5. 总结
    # ============================================================
    print(f"\n{'='*78}")
    print("总结")
    print(f"{'='*78}")
    print(f"""
  算法: KDE（核密度估计）
  振动阈值: {threshold:.4f}
  清洗移除: {removed} 样本
  有效样本: {is_valid.sum()}

  使用方法:
    当振动均值 >= {threshold:.2f} 时，判定电梯在运行
    当振动均值 <  {threshold:.2f} 时，判定电梯已停机

  注意:
    - 阈值仅从振动数据分布求得（KDE 密度谷底）
    - 不需要电流数据
    - 建议定期用新数据重新计算阈值
""")
