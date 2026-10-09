"""
扶梯振动阈值自动确定 — KDE算法 + Athena直连 + 电流验证
==================================================
从 AWS Athena 直接查询数据，用 KDE 求振动阈值，电流做验证
不生成中间 CSV 文件，数据直接传入算法

依赖：numpy, pandas, scikit-learn, pyathena, boto3
安装：pip install numpy pandas scikit-learn pyathena boto3

运行：python kde_athena.py
"""

import numpy as np
import pandas as pd
import warnings

from sklearn.neighbors import KernelDensity


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
# KDE 核密度估计算法
# ============================================================

def kde_threshold(data: np.ndarray, bandwidth: str = "scott",
                   recall_bias: float = 0.0) -> float:
    """
    KDE：估计概率密度曲线，找两峰之间的密度谷底作为阈值

    原理：
      1. 用核函数（高斯核）对数据做平滑密度估计
      2. 在密度曲线上找所有局部极大值（峰）
      3. 取密度最高的两个峰（停机峰 + 运行峰）
      4. 在两峰之间找密度最低的谷底，即为阈值
      5. 可选：通过 recall_bias 将阈值向停机峰方向偏移，提高覆盖率

    参数:
        data        : 一维振动数据数组
        bandwidth   : 核密度带宽，"scott" 为自动选择
        recall_bias : 覆盖率偏移系数（0.0~1.0）
                      0.0  = 原始谷底（默认，无偏移）
                      0.1  = 向停机峰方向移动 10% 的两峰间距
                      0.2  = 向停机峰方向移动 20% 的两峰间距
                      值越大 → 阈值越低 → 覆盖率越高（但准确率可能下降）

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
    valley_x = valleys[0][0]

    # ★ 覆盖率偏移：将阈值向停机峰方向移动
    # 停机峰 = lo（两峰中较小值），运行峰 = hi
    # bias > 0 时，阈值从谷底向 lo 方向移动，降低阈值 → 更多数据被判为运行 → 覆盖率提高
    if recall_bias > 0:
        shift = (valley_x - lo) * recall_bias
        adjusted = valley_x - shift
        return float(adjusted)

    return float(valley_x)


def kde_threshold_f1_optimized(data: np.ndarray, current: np.ndarray,
                                current_threshold: float,
                                bandwidth: str = "scott",
                                search_range: float = 0.3) -> float:
    """
    F1 优化版 KDE 阈值：在 KDE 谷底附近搜索使 F1 最高的阈值

    原理：
      1. 先用标准 KDE 求出谷底阈值
      2. 在谷底两侧 ±search_range * 两峰间距 范围内遍历候选阈值
      3. 对每个候选阈值用电流算 F1，取 F1 最高的

    适用条件：需要有电流数据做验证

    参数:
        data              : 振动数据
        current           : 电流数据（与振动等长）
        current_threshold : 电流分界点
        bandwidth         : KDE 带宽
        search_range      : 搜索范围比例（0.3 = 两峰间距的 30%）

    返回:
        best_threshold : F1 最高的阈值
    """
    # 对齐长度：如果 data 和 current 长度不同，取较短的那个
    min_len = min(len(data), len(current))
    data = data[:min_len]
    current = current[:min_len]

    # 同时过滤 NaN
    valid_mask = np.isfinite(data) & np.isfinite(current)
    data = data[valid_mask]
    current = current[valid_mask]

    if len(data) < 20:
        raise ValueError("有效数据太少")

    # 先求标准 KDE 谷底
    base_threshold = kde_threshold(data, bandwidth=bandwidth)

    # 求两峰位置（确定搜索范围）
    x = data.reshape(-1, 1)
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(x)
    grid = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
    density = np.exp(kde.score_samples(grid))
    grid_x = grid[:, 0]

    peaks = []
    for i in range(1, len(density) - 1):
        if density[i] > density[i - 1] and density[i] >= density[i + 1]:
            peaks.append((grid_x[i], density[i]))

    if len(peaks) < 2:
        return base_threshold

    peaks.sort(key=lambda p: p[1], reverse=True)
    peak_lo = min(peaks[0][0], peaks[1][0])
    peak_hi = max(peaks[0][0], peaks[1][0])
    peak_gap = peak_hi - peak_lo

    # 在谷底两侧搜索
    search_lo = base_threshold - search_range * peak_gap
    search_hi = base_threshold + search_range * peak_gap
    candidates = np.linspace(search_lo, search_hi, 50)

    true_running = current > current_threshold

    best_f1 = -1
    best_t = base_threshold

    for t in candidates:
        pred_running = data > t
        tp = np.sum(pred_running & true_running)
        fp = np.sum(pred_running & ~true_running)
        fn = np.sum(~pred_running & true_running)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        if f1 >= best_f1:
            best_f1 = f1
            best_t = t

    return float(best_t)


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
# 评估函数（用电流做真值）
# ============================================================

def evaluate(vibration: np.ndarray, current: np.ndarray,
             current_threshold: float, vib_threshold: float) -> dict:
    """以电流判断为准真值，评估振动阈值的准确率/覆盖率"""
    true_running = current > current_threshold
    pred_running = vibration > vib_threshold

    tp = int(np.sum(pred_running & true_running))
    fp = int(np.sum(pred_running & ~true_running))
    fn = int(np.sum(~pred_running & true_running))
    tn = int(np.sum(~pred_running & ~true_running))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision, "recall": recall, "f1_score": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ============================================================
# 评估函数（用人工阈值做真值）
# ============================================================

def evaluate_manual(vibration: np.ndarray,
                   manual_threshold: float,
                   kde_threshold: float) -> dict:
    """
    以人工指定的阈值为标准真值，评估 KDE 阈值的准确率/覆盖率

    逻辑:
      振动 > 人工阈值 → 真值=运行
      振动 > KDE阈值 → 预测=运行
      对比两者，算 TP/FP/FN/TN
    """
    true_running = vibration > manual_threshold
    pred_running = vibration > kde_threshold

    tp = int(np.sum(pred_running & true_running))
    fp = int(np.sum(pred_running & ~true_running))
    fn = int(np.sum(~pred_running & true_running))
    tn = int(np.sum(~pred_running & ~true_running))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision, "recall": recall, "f1_score": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ============================================================
# 电流分界点自动检测（KDE）
# ============================================================

def find_current_threshold_kde(current: np.ndarray,
                                equipment_ids: np.ndarray = None) -> float:
    """
    用 KDE 自动检测电流数据的分界点

    如果有多台设备且电流范围差异大，按设备分别检测，
    取各设备分界点的中位数作为全局分界点
    """
    current = np.asarray(current, dtype=np.float64)
    cur_clean = current[np.isfinite(current)]

    if len(cur_clean) < 10:
        raise ValueError("电流数据太少")

    cmin, cmax = cur_clean.min(), cur_clean.max()
    if cmax - cmin < 0.01:
        raise ValueError("电流数据几乎全相同，无法检测分界点")

    # 多设备：检查是否需要按设备分别检测
    if equipment_ids is not None:
        eq_arr = np.asarray(equipment_ids)
        unique_eqs = np.unique(eq_arr[np.isfinite(current)])
        if len(unique_eqs) > 1:
            eq_maxes = {}
            for eq in unique_eqs:
                eq_cur = current[eq_arr == eq]
                eq_cur = eq_cur[np.isfinite(eq_cur)]
                if len(eq_cur) > 0:
                    eq_maxes[str(eq)] = eq_cur.max()

            max_values = list(eq_maxes.values())
            if max(max_values) / (min(max_values) + 1e-9) > 2:
                print(f"  检测到 {len(unique_eqs)} 台设备，"
                      f"电流范围差异大，按设备分别检测:")

                eq_thresholds = []
                for eq in unique_eqs:
                    eq_mask = eq_arr == eq
                    eq_cur = current[eq_mask]
                    eq_cur = eq_cur[np.isfinite(eq_cur)]
                    if len(eq_cur) < 10 or eq_cur.max() - eq_cur.min() < 0.01:
                        print(f"    设备 {eq}: 电流范围太小，跳过")
                        continue
                    # 电流不做 MAD 清洗（会误删停机段）
                    try:
                        t = kde_threshold(eq_cur)
                        eq_thresholds.append(t)
                        n_r = int(np.sum(eq_cur > t))
                        n_s = int(np.sum(eq_cur <= t))
                        print(f"    设备 {eq}: 范围[{eq_cur.min():.2f}, "
                              f"{eq_cur.max():.2f}]  分界点={t:.4f}  "
                              f"运行{n_r}条/停机{n_s}条")
                    except Exception as e:
                        print(f"    设备 {eq}: KDE失败({e})，跳过")

                if len(eq_thresholds) == 0:
                    raise ValueError("所有设备的电流都无法检测分界点")

                best_t = float(np.median(eq_thresholds))
                print(f"\n  各设备分界点: "
                      f"{[f'{t:.4f}' for t in eq_thresholds]}")
                print(f"  -> 全局电流分界点(中位数): {best_t:.4f}")
                return best_t

    # 单设备：整体检测
    return kde_threshold(cur_clean)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # ★★★ 修改这里：改成你要查询的设备和日期范围 ★★★
    # ============================================================
    EQUIPMENT_NUMBERS = ['30248586']          # 设备号列表，可多台
    START_DATE = '2026-06-10'                 # 开始日期
    END_DATE = '2026-06-15'                   # 结束日期

    # 振动列名
    VIBRATION_COL = 'gbvibforwardrms'
    # 电流列名（如果没有电流数据，设为 None）
    CURRENT_COLS = ['motorcurrent1avg', 'motorcurrent2avg', 'motorcurrent3avg']
    # 电流阈值：设为 None 表示自动检测（推荐），也可以手动指定如 2.0
    CURRENT_THRESHOLD = None

    # ★★★ 人工标准阈值：你人工看数据定的阈值，用来验证 KDE 阈值的准确率/覆盖率 ★★★
    # 设为 None 表示不使用人工标准；设为一个数字如 5.0 表示用 5.0 作为标准
    MANUAL_THRESHOLD = 3

    # ★★★ 覆盖率优化模式 ★★★
    # "none"    = 原始 KDE 谷底（默认，不优化）
    # "bias"    = 阈值偏移法：将阈值向停机峰方向移动，提高覆盖率（不需要电流）
    # "f1"      = F1 优化法：在谷底附近搜索使 F1 最高的阈值（需要电流数据）
    # "auto"    = 自动选择：有电流用 "f1"，没电流用 "bias"
    OPTIMIZE_MODE = "auto"

    # ★★★ 偏移量（仅 "bias" 模式生效）★★★
    # 0.0 = 无偏移；0.1 = 向停机峰移动 10% 两峰间距；0.2 = 移动 20%
    # 值越大 → 阈值越低 → 覆盖率越高（但准确率可能下降）
    RECALL_BIAS = 0.15

    # ============================================================
    # 1. 从 Athena 查询数据（不生成 CSV，直接用）
    # ============================================================
    print("=" * 78)
    print("扶梯振动阈值自动确定 — KDE算法 + Athena直连 + 电流验证")
    print("=" * 78)

    # 构建 SQL
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

    print(f"\n查询设备: {EQUIPMENT_NUMBERS}")
    print(f"日期范围: {START_DATE} ~ {END_DATE}")

    conn = Athena_Connector()
    try:
        combined = conn.query(sql)
    finally:
        conn.close()

    print(f"查询到 {len(combined)} 条数据")
    print(f"列名: {list(combined.columns)}")

    if len(combined) == 0:
        print("\n[错误] 没有查询到数据！请检查设备号和日期范围。")
        exit(1)

    # 去重
    dedup_cols = [c for c in ['equipmentnumber', 'event_time_shanghai']
                  if c in combined.columns]
    if len(dedup_cols) == 2:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
        if before != len(combined):
            print(f"去重: {before} -> {len(combined)} 条")

    # ============================================================
    # 2. 提取振动数据
    # ============================================================
    if VIBRATION_COL not in combined.columns:
        print(f"\n[错误] 数据中没有找到振动列 '{VIBRATION_COL}'！")
        print(f"  列名: {list(combined.columns)}")
        exit(1)

    vibration = combined[VIBRATION_COL].values.astype(float)

    # ============================================================
    # 3. 提取电流数据（可选）
    # ============================================================
    has_current = all(c in combined.columns for c in CURRENT_COLS) if CURRENT_COLS else False
    current_mean = None

    if has_current:
        c1 = combined[CURRENT_COLS[0]].values.astype(float)
        c2 = combined[CURRENT_COLS[1]].values.astype(float)
        c3 = combined[CURRENT_COLS[2]].values.astype(float)
        current_mean = (c1 + c2 + c3) / 3.0
        valid_cur = current_mean[np.isfinite(current_mean)]

        print(f"\n电流数据: 可用")

        if len(valid_cur) < 10 or valid_cur.max() - valid_cur.min() < 0.01:
            print(f"  警告: 电流数据太少或几乎全相同，无法用于验证")
            has_current = False
            current_mean = None
        elif CURRENT_THRESHOLD is None:
            print(f"\n  [自动检测电流分界点 — KDE]")
            try:
                eq_ids = (combined['equipmentnumber'].values
                          if 'equipmentnumber' in combined.columns else None)
                CURRENT_THRESHOLD = find_current_threshold_kde(
                    current_mean, equipment_ids=eq_ids)
                print(f"\n  -> 最终电流分界点: {CURRENT_THRESHOLD:.4f}")
            except Exception as e:
                print(f"  电流分界点检测失败: {e}")
                has_current = False
                current_mean = None
        else:
            n_run = int(np.sum(valid_cur > CURRENT_THRESHOLD))
            n_stop = int(np.sum(valid_cur <= CURRENT_THRESHOLD))
            print(f"  使用手动电流阈值: {CURRENT_THRESHOLD}")
            print(f"  电流 > 阈值（运行）: {n_run} 条")
            print(f"  电流 <= 阈值（停机）: {n_stop} 条")
            if n_run == 0:
                print(f"  警告: 没有电流 > {CURRENT_THRESHOLD} 的数据！")
                has_current = False
                current_mean = None
    else:
        print(f"\n电流数据: 不可用")
        print(f"  -> 阈值仅从振动分布求得，不验证准确率")

    # ============================================================
    # 4. 振动数据统计
    # ============================================================
    valid_vib = vibration[np.isfinite(vibration)]
    print(f"\n振动数据统计:")
    print(f"  有效样本: {len(valid_vib)}")
    print(f"  范围: [{valid_vib.min():.3f}, {valid_vib.max():.3f}]")
    print(f"  均值: {valid_vib.mean():.3f}, 中位数: {np.median(valid_vib):.3f}")

    if 'equipmentnumber' in combined.columns:
        print(f"  设备列表: {combined['equipmentnumber'].unique()}")

    # ============================================================
    # 5. KDE 求振动阈值
    # ============================================================
    print(f"\n{'='*78}")
    print("KDE 振动阈值分析")
    print(f"{'='*78}")

    cleaned, is_valid = clean_vibration(vibration)
    removed = len(vibration) - is_valid.sum()
    print(f"  [清洗] 移除 {removed} 个异常样本，保留 {is_valid.sum()} 个")

    # 先求标准 KDE 谷底阈值（作为基准）
    raw_threshold = kde_threshold(cleaned)
    raw_margin = analyze_margin(cleaned, raw_threshold)

    print(f"\n  [标准 KDE 谷底]")
    print(f"    原始阈值:   {raw_threshold:.4f}")
    if "error" not in raw_margin:
        print(f"    停机 P99:   {raw_margin['stop_p99']:.3f}")
        print(f"    运行 P1:    {raw_margin['run_p1']:.3f}")
        print(f"    最小裕度:   {raw_margin['min_margin']:+.3f}")
        print(f"    无人区宽度: {raw_margin['gap_width']:.3f}")

    # 根据优化模式选择最终阈值
    mode = OPTIMIZE_MODE
    if mode == "auto":
        mode = "f1" if (has_current and current_mean is not None) else "bias"

    if mode == "none":
        threshold = raw_threshold
        print(f"\n  [优化模式] none — 使用原始谷底，不优化")

    elif mode == "bias":
        threshold = kde_threshold(cleaned, recall_bias=RECALL_BIAS)
        shift = raw_threshold - threshold
        print(f"\n  [优化模式] bias — 向停机峰偏移 {RECALL_BIAS*100:.0f}% 两峰间距")
        print(f"    偏移前: {raw_threshold:.4f}")
        print(f"    偏移后: {threshold:.4f}  (降低 {shift:.4f})")

    elif mode == "f1":
        if has_current and current_mean is not None:
            # 确保电流数据用同样的 is_valid 掩码过滤
            cur_cleaned = current_mean[is_valid]
            # 额外保护：如果长度仍不一致，函数内部会自动对齐
            threshold = kde_threshold_f1_optimized(
                cleaned, cur_cleaned, CURRENT_THRESHOLD)
            shift = raw_threshold - threshold
            print(f"\n  [优化模式] f1 — F1 最大化搜索（有电流验证）")
            print(f"    搜索前: {raw_threshold:.4f}")
            print(f"    搜索后: {threshold:.4f}  (变化 {shift:+.4f})")
        else:
            threshold = raw_threshold
            print(f"\n  [优化模式] f1 — 无电流数据，回退到原始谷底")

    margin = analyze_margin(cleaned, threshold)
    print(f"\n  [最终阈值]")
    print(f"    阈值:       {threshold:.4f}")
    if "error" not in margin:
        print(f"    停机 P99:   {margin['stop_p99']:.3f}")
        print(f"    运行 P1:    {margin['run_p1']:.3f}")
        print(f"    到停机裕度: {margin['margin_to_stop']:+.3f}")
        print(f"    到运行裕度: {margin['margin_to_run']:+.3f}")
        print(f"    最小裕度:   {margin['min_margin']:+.3f}")
        print(f"    无人区宽度: {margin['gap_width']:.3f}")

    # ============================================================
    # 6. 电流验证（如果可用）
    # ============================================================
    if has_current and current_mean is not None:
        print(f"\n{'='*78}")
        print("电流验证")
        print(f"{'='*78}")

        valid_vib_arr = vibration[is_valid]
        valid_cur_arr = current_mean[is_valid]

        m = evaluate(valid_vib_arr, valid_cur_arr,
                     CURRENT_THRESHOLD, threshold)
        print(f"\n  KDE 阈值 {threshold:.4f} 的验证结果:")
        print(f"    准确率 (Precision): {m['precision']:.4f}")
        print(f"    覆盖率 (Recall):    {m['recall']:.4f}")
        print(f"    F1-score:           {m['f1_score']:.4f}")
        print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

    # ============================================================
    # 6b. 人工阈值验证（如果设置了人工标准阈值）
    # ============================================================
    if MANUAL_THRESHOLD is not None:
        print(f"\n{'='*78}")
        print("人工标准阈值验证")
        print(f"{'='*78}")

        valid_vib_arr = vibration[is_valid]
        m = evaluate_manual(valid_vib_arr, MANUAL_THRESHOLD, threshold)

        n_manual_run = int(np.sum(valid_vib_arr > MANUAL_THRESHOLD))
        n_manual_stop = int(np.sum(valid_vib_arr <= MANUAL_THRESHOLD))
        print(f"\n  人工标准阈值: {MANUAL_THRESHOLD}")
        print(f"  KDE 阈值:     {threshold:.4f}")
        print(f"  差值:         {abs(threshold - MANUAL_THRESHOLD):.4f}")
        print(f"  人工标准判定: 运行 {n_manual_run} 条 / 停机 {n_manual_stop} 条")
        print(f"\n  KDE 阈值 {threshold:.4f} vs 人工阈值 {MANUAL_THRESHOLD} 的验证结果:")
        print(f"    准确率 (Precision): {m['precision']:.4f}")
        print(f"    覆盖率 (Recall):    {m['recall']:.4f}")
        print(f"    F1-score:           {m['f1_score']:.4f}")
        print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

        if m['precision'] < 0.95 or m['recall'] < 0.95:
            if threshold < MANUAL_THRESHOLD:
                print(f"\n  提示: KDE 阈值({threshold:.2f}) < 人工阈值({MANUAL_THRESHOLD:.2f})")
                print(f"    KDE 偏低，会把更多人判定为运行 → 准确率偏低，覆盖率高")
                print(f"    如需提高准确率，可手动调高 KDE 阈值")
            else:
                print(f"\n  提示: KDE 阈值({threshold:.2f}) > 人工阈值({MANUAL_THRESHOLD:.2f})")
                print(f"    KDE 偏高，会把更少人判定为运行 → 准确率高，覆盖率偏低")
                print(f"    如需提高覆盖率，可手动调低 KDE 阈值")
        else:
            print(f"\n  KDE 阈值与人工标准高度一致，效果良好。")
    # ============================================================
    # 7. 逐设备分析
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
    # 8. 总结
    # ============================================================
    print(f"\n{'='*78}")
    print("总结")
    print(f"{'='*78}")
    print(f"""
  算法: KDE（核密度估计）
  数据来源: Athena 直连（设备 {EQUIPMENT_NUMBERS}，{START_DATE} ~ {END_DATE}）
  振动阈值: {threshold:.4f}
  清洗移除: {removed} 样本
  有效样本: {is_valid.sum()}

  使用方法:
    当振动均值 >= {threshold:.2f} 时，判定电梯在运行
    当振动均值 <  {threshold:.2f} 时，判定电梯已停机

  注意:
    - 阈值仅从振动数据分布求得（KDE 密度谷底）
    - 电流仅用于验证，不影响阈值计算
    - 建议定期用新数据重新计算阈值
""")
