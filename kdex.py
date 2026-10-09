"""
扶梯振动阈值自动确定 — KDE算法 + Athena直连 + 速度验证
==================================================
从 AWS Athena 直接查询数据，用 KDE 求振动阈值，速度做验证
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

def kde_threshold(data: np.ndarray, bandwidth: str = "scott") -> float:
    """
    KDE：估计概率密度曲线，找两峰之间的密度谷底作为阈值

    原理：
      1. 用核函数（高斯核）对数据做平滑密度估计
      2. 在密度曲线上找所有局部极大值（峰）
      3. 取密度最高的两个峰（停机峰 + 运行峰）
      4. 在两峰之间找密度最低的谷底，即为阈值

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
# 评估函数（用速度做真值）
# ============================================================

def evaluate_by_speed(vibration: np.ndarray, speed: np.ndarray,
                      vib_threshold: float) -> dict:
    """
    以速度判断为准真值，评估振动阈值的准确率/覆盖率

    速度为 0 → 停机，速度不为 0 → 运行
    排除速度或振动为 NaN/Inf 的数据点
    """
    valid = np.isfinite(speed) & np.isfinite(vibration)
    vib_v = vibration[valid]
    spd_v = speed[valid]

    true_running = spd_v != 0
    pred_running = vib_v > vib_threshold

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
# 主程序
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # ★★★ 修改这里：改成你要查询的设备和日期范围 ★★★
    # ============================================================
    EQUIPMENT_NUMBERS = ['30377926']          # 设备号列表，可多台
    START_DATE = '2026-07-20'                 # 开始日期（总范围起点）
    END_DATE = '2026-07-26'                   # 结束日期（总范围终点）

    # ★★★ 训练/测试分界日期 ★★★
    # <= SPLIT_DATE 的数据用来求阈值（训练集）
    # >  SPLIT_DATE 的数据用来验证阈值（测试集）
    # 设为 None 表示不分训练/测试，全部数据一起求阈值+验证
    SPLIT_DATE = '2026-07-23'                         # 如 '2026-07-11'

    # 振动列名
    VIBRATION_COL = 'gbvibforwardrms'
  # 速度列名（速度为0=停机，速度不为0=运行，用于验证）
    SPEED_COL = 'stepbandspeedleftavg'

    # ★★★ 人工标准阈值：你人工看数据定的阈值，用来验证 KDE 阈值的准确率/覆盖率 ★★★
    # 设为 None 表示不使用人工标准；设为一个数字如 5.0 表示用 5.0 作为标准
    MANUAL_THRESHOLD = 2.5

    # ============================================================
    # 1. 从 Athena 查询数据（不生成 CSV，直接用）
    # ============================================================
    # 构建 SQL
    eq_list = ", ".join(f"'{e}'" for e in EQUIPMENT_NUMBERS)
    sql = f"""
        SELECT equipmentnumber, gbvibforwardrms, motorcurrent1avg, stepbandspeedleftavg,  modeset, operationstatus,
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

    if len(combined) == 0:
        print("\n[错误] 没有查询到数据！请检查设备号和日期范围。")
        exit(1)

    # 去重
    dedup_cols = [c for c in ['equipmentnumber', 'event_time_shanghai']
                  if c in combined.columns]
    if len(dedup_cols) == 2:
        combined = combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    # ============================================================
    # 2. 训练/测试集划分（按日期切分）
    # ============================================================
    use_split = (SPLIT_DATE is not None
                 and 'event_time_shanghai' in combined.columns)

    if use_split:
        combined['date_str'] = combined['event_time_shanghai'].astype(str).str[:10]
        train_df = combined[combined['date_str'] <= SPLIT_DATE].copy()
        test_df = combined[combined['date_str'] > SPLIT_DATE].copy()

        if len(train_df) < 20:
            print("[错误] 训练集数据太少，请调整 SPLIT_DATE")
            exit(1)
        if len(test_df) < 20:
            print("[警告] 测试集数据太少，验证结果可能不可靠")
    else:
        train_df = combined
        test_df = None

    # ============================================================
    # 3. 提取振动数据（训练集）
    # ============================================================
    if VIBRATION_COL not in train_df.columns:
        print(f"\n[错误] 数据中没有找到振动列 '{VIBRATION_COL}'！")
        print(f"  列名: {list(train_df.columns)}")
        exit(1)

    vibration_train = train_df[VIBRATION_COL].values.astype(float)

    # ============================================================
    # 4. 提取速度数据（可选，训练集）
    # ============================================================
    has_speed = SPEED_COL in train_df.columns if SPEED_COL else False
    speed_train = None

    if has_speed:
        speed_train = train_df[SPEED_COL].values.astype(float)
        valid_spd = speed_train[np.isfinite(speed_train)]

        if len(valid_spd) < 10:
            has_speed = False
            speed_train = None
        else:
            n_run = int(np.sum(valid_spd != 0))
            n_stop = int(np.sum(valid_spd == 0))
            if n_run == 0:
                has_speed = False
                speed_train = None

    # ============================================================
    # 5. 振动数据统计（训练集）
    # ============================================================
    valid_vib = vibration_train[np.isfinite(vibration_train)]

    # ============================================================
    # 6. KDE 求振动阈值（用训练集）
    # ============================================================
    cleaned_train, is_valid_train = clean_vibration(vibration_train)
    removed = len(vibration_train) - is_valid_train.sum()

    threshold = kde_threshold(cleaned_train)
    margin = analyze_margin(cleaned_train, threshold)

    # ============================================================
    # 7. 验证：训练集内部 + 测试集（如果可用）
    # ============================================================

    # --- 7a. 训练集内部验证 ---
    vib_train_valid = vibration_train[is_valid_train]

    # 速度验证
    if has_speed and speed_train is not None:
        spd_train_valid = speed_train[is_valid_train]
        m_train = evaluate_by_speed(vib_train_valid, spd_train_valid, threshold)

    # 人工阈值验证
    if MANUAL_THRESHOLD is not None:
        m_manual_train = evaluate_manual(vib_train_valid, MANUAL_THRESHOLD, threshold)

    # --- 7b. 测试集验证（核心：用训练集求出的阈值验证测试集数据）---
    if use_split and test_df is not None and len(test_df) >= 20:
        vibration_test = test_df[VIBRATION_COL].values.astype(float)
        cleaned_test, is_valid_test = clean_vibration(vibration_test)
        removed_test = len(vibration_test) - is_valid_test.sum()

        vib_test_valid = vibration_test[is_valid_test]

        # 测试集裕度分析
        margin_test = analyze_margin(cleaned_test, threshold)

        # 测试集速度验证
        m_test = None
        if has_speed and SPEED_COL in test_df.columns:
            speed_test = test_df[SPEED_COL].values.astype(float)
            spd_test_valid = speed_test[is_valid_test]
            m_test = evaluate_by_speed(vib_test_valid, spd_test_valid, threshold)

        # 测试集人工阈值验证
        m_manual_test = None
        if MANUAL_THRESHOLD is not None:
            m_manual_test = evaluate_manual(vib_test_valid, MANUAL_THRESHOLD, threshold)
    else:
        removed_test = 0
        is_valid_test = None
        m_test = None
        m_manual_test = None

    # ============================================================
    # 8. 总结 + 测试集对比
    # ============================================================
    print("=" * 78)
    print("总结")
    print("=" * 78)

    if use_split:
        print(f"""
  算法: KDE（核密度估计）
  数据来源: Athena 直连（设备 {EQUIPMENT_NUMBERS}，{START_DATE} ~ {END_DATE}）
  训练集: {len(train_df)} 条（{START_DATE} ~ {SPLIT_DATE}）→ 用来求阈值
  测试集: {len(test_df)} 条（{SPLIT_DATE} ~ {END_DATE}）→ 用来验证阈值
  振动阈值: {threshold:.4f}
  清洗移除: 训练集 {removed} / 测试集 {removed_test}
  有效样本: 训练集 {is_valid_train.sum()} / 测试集 {is_valid_test.sum() if is_valid_test is not None else 0}
""")

        # 测试集对比：人工阈值 vs 速度验证
        print()
        print("=" * 78)
        print("测试集对比 — 人工阈值 vs 速度验证")
        print("=" * 78)

        if m_test is None:
            print("  [警告] 速度验证数据不可用（速度字段缺失或全为0），无法对比")
        elif m_manual_test is None:
            print("  [警告] 人工阈值未设置（MANUAL_THRESHOLD=None），无法对比")
        else:
            print(f"  {'指标':<20} {'人工阈值':<12} {'速度验证':<12} {'差值':<10}")
            print("  " + "-" * 52)
            print(f"  {'准确率':<20} {m_manual_test['precision']:<12.4f} "
                  f"{m_test['precision']:<12.4f} "
                  f"{m_manual_test['precision']-m_test['precision']:+.4f}")
            print(f"  {'覆盖率':<20} {m_manual_test['recall']:<12.4f} "
                  f"{m_test['recall']:<12.4f} "
                  f"{m_manual_test['recall']-m_test['recall']:+.4f}")
            print(f"  {'F1':<20} {m_manual_test['f1_score']:<12.4f} "
                  f"{m_test['f1_score']:<12.4f} "
                  f"{m_manual_test['f1_score']-m_test['f1_score']:+.4f}")
    else:
        print(f"""
  算法: KDE（核密度估计）
  数据来源: Athena 直连（设备 {EQUIPMENT_NUMBERS}，{START_DATE} ~ {END_DATE}）
  振动阈值: {threshold:.4f}
  清洗移除: {removed} 样本
  有效样本: {is_valid_train.sum()}
""")
