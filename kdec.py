"""
扶梯振动阈值自动确定 — KDE算法 + Athena直连 + 速度验证
==================================================
从 AWS Athena 直接查询数据，用 KDE 求振动阈值，速度做验证
不生成中间 CSV 文件，数据直接传入算法
依赖：numpy, pandas, scikit-learn, pyathena, boto3
安装：pip install numpy pandas scikit-learn pyathena boto3

运行：python kdez.py
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
                    percentile_clip: float = 0.1) -> tuple:
    """
    振动数据清洗：去除异常值

    策略:
      1. 去除 NaN / Inf
      2. 百分位截断：去掉最高和最低 0.1% 的极端值

    注意：
      不使用 MAD 离群点检测。原因是扶梯振动数据是双峰分布
      （停机簇 + 运行簇），当停机数据占比远大于运行数据时，
      MAD 会以停机簇为基准，把运行数据全部当成离群点删掉，
      导致 KDE 找不到运行峰，阈值失效。

    返回:
        cleaned : 清洗后的数组
        is_valid: 布尔掩码（与原数组等长）
    """
    vibration = np.asarray(vibration, dtype=np.float64)
    n = len(vibration)
    is_valid = np.ones(n, dtype=bool)

    # 1. 去除 NaN / Inf
    is_valid &= np.isfinite(vibration)
    if is_valid.sum() < 10:
        return vibration[is_valid], is_valid

    # 2. 百分位截断：去掉最高和最低 percentile_clip% 的极端值
    v = vibration[is_valid]
    p_low = np.percentile(v, percentile_clip)
    p_high = np.percentile(v, 100 - percentile_clip)
    is_valid[is_valid] &= (v >= p_low) & (v <= p_high)

    cleaned = vibration[is_valid]
    return cleaned, is_valid


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
    EQUIPMENT_NUMBERS = ['30536165']          # 设备号列表，可多台
    START_DATE = '2026-07-20'                 # 开始日期（总范围起点）
    END_DATE = '2026-07-26'                   # 结束日期（总范围终点）

    SPLIT_DATE = '2026-07-23'

    # 振动列名
    VIBRATION_COL = 'gbvibforwardrms'
    # 速度列名（速度为0=停机，速度不为0=运行，用于验证）
    SPEED_COL = 'stepbandspeedleftavg'
    # ★★★ 人工标准阈值：你人工看数据定的阈值，用来验证 KDE 阈值的准确率/覆盖率 ★★★
    # 设为 None 表示不使用人工标准；设为一个数字如 5.0 表示用 5.0 作为标准
    MANUAL_THRESHOLD = 4

    # ============================================================
    # 1. 从 Athena 查询数据
    # ============================================================
    eq_list = ", ".join(f"'{e}'" for e in EQUIPMENT_NUMBERS)
    sql = f"""
        SELECT equipmentnumber, gbvibforwardrms, motorcurrent1avg, stepbandspeedleftavg, 
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
        print("[错误] 没有查询到数据！请检查设备号和日期范围。")
        exit(1)

    # 去重
    dedup_cols = [c for c in ['equipmentnumber', 'event_time_shanghai']
                  if c in combined.columns]
    if len(dedup_cols) == 2:
        combined = combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    # ============================================================
    # 2. 测试/验证集划分（按日期切分）
    # ============================================================
    combined['date_str'] = combined['event_time_shanghai'].astype(str).str[:10]
    test_df = combined[combined['date_str'] <= SPLIT_DATE].copy()      # 测试集：求阈值
    val_df = combined[combined['date_str'] > SPLIT_DATE].copy()        # 验证集：验证阈值

    if len(test_df) < 20:
        print("[错误] 测试集数据太少，请调整 SPLIT_DATE")
        exit(1)

    # ============================================================
    # 3. 测试集：清洗 + KDE 求阈值（只算不打印细节）
    # ============================================================
    vibration_test = test_df[VIBRATION_COL].values.astype(float)
    cleaned_test, is_valid_test = clean_vibration(vibration_test)
    threshold = kde_threshold(cleaned_test)

    # ============================================================
    # 4. 验证集：清洗 + 速度验证 + 人工阈值验证
    # ============================================================
    vibration_val = val_df[VIBRATION_COL].values.astype(float)
    cleaned_val, is_valid_val = clean_vibration(vibration_val)
    removed_val = len(vibration_val) - is_valid_val.sum()

    # 速度数据
    has_speed = SPEED_COL in val_df.columns
    speed_val = val_df[SPEED_COL].values.astype(float) if has_speed else None

    # 验证用原始数据（只去 NaN/Inf），不用清洗掩码
    eval_mask = np.isfinite(vibration_val)
    if has_speed:
        eval_mask &= np.isfinite(speed_val)
    vib_val_raw = vibration_val[eval_mask]
    spd_val_raw = speed_val[eval_mask] if has_speed else None

    # 速度验证
    m_speed = evaluate_by_speed(vib_val_raw, spd_val_raw, threshold) if has_speed else None

    # 人工阈值验证
    m_manual = evaluate_manual(vib_val_raw, MANUAL_THRESHOLD, threshold) if MANUAL_THRESHOLD is not None else None

    # ============================================================
    # 5. 输出
    # ============================================================
    print("=" * 60)
    print(f"设备号: {EQUIPMENT_NUMBERS}")
    print(f"日期范围: {START_DATE} ~ {END_DATE}（分界: {SPLIT_DATE}）")
    print(f"  测试集: {len(test_df)} 条 → 求阈值")
    print(f"  验证集: {len(val_df)} 条 → 验证阈值")
    print("=" * 60)

    # 速度可用数据
    if has_speed:
        valid_spd = speed_val[np.isfinite(speed_val)]
        n_run = int(np.sum(valid_spd != 0))
        n_stop = int(np.sum(valid_spd == 0))
        print(f"\n速度数据: 可用")
        print(f"  速度 != 0（运行）: {n_run} 条")
        print(f"  速度 == 0（停机）: {n_stop} 条")
    else:
        print(f"\n速度数据: 不可用")

    # 验证集清洗情况
    print(f"\n验证集清洗: 移除 {removed_val} 个，保留 {is_valid_val.sum()} 个")

    # 算法阈值
    print(f"\nKDE 振动阈值: {threshold:.4f}")

    # 验证集速度验证结果
    if m_speed is not None:
        print(f"\n速度验证（验证集）:")
        print(f"  准确率 (Precision): {m_speed['precision']:.4f}")
        print(f"  覆盖率 (Recall):    {m_speed['recall']:.4f}")
        print(f"  F1-score:           {m_speed['f1_score']:.4f}")

    # 验证集人工阈值验证结果
    if m_manual is not None:
        print(f"\n人工阈值验证（验证集）:")
        print(f"  人工阈值: {MANUAL_THRESHOLD}  KDE阈值: {threshold:.4f}")
        print(f"  准确率 (Precision): {m_manual['precision']:.4f}")
        print(f"  覆盖率 (Recall):    {m_manual['recall']:.4f}")
        print(f"  F1-score:           {m_manual['f1_score']:.4f}")

    print(f"\n{'=' * 60}")
