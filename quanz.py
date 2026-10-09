"""
Athena 数据查询(去除空的数据+负 空的速度) 
        FROM data_cleansed."anyescalator"  
  1. 速度负值/无效值处理 —— 只有 speed > SPEED_MIN_THRESHOLD 才算运行
  2. 使用 KDE 计算振动自动阈值，KDE 失败时使用 Otsu 算法
  3. 使用速度运行状态作为参考，验证 KDE 阈值和人工阈值
  4. 速度数据质量检查 —— 如果速度数据全是同一个值或全为负，自动告警并跳过速度验证
  5. 无效速度数据不参与评估计算，避免拉低召回率
  6. 输出速度数据质量报告 保存在git中的文件 csv+图片
"""

import os
import warnings
import boto3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyathena
from sklearn.neighbors import KernelDensity

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

EQUIPMENT_NUMBERS = ['45036936']
START_DATE = '2026-09-01'
END_DATE = '2026-09-15'

SPLIT_DATE = '2026-09-01'          # <= 求阈值，> 验证

MANUAL_THRESHOLD = 4.0             # 人工阈值，设为 None 则不验证
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
# ---- 速度相关配置（新增）----
SPEED_MIN_THRESHOLD = 0          # 速度大于此值才算"运行中"（避免 -1、0.001 等无效值误判）
SPEED_VALID_RATIO = 0.0001           # 有效速度样本占比低于此值时，认为速度数据不可靠，跳过速度验证
SPEED_UNIQUE_MIN = 1               # 速度唯一值数量低于此值时，认为速度数据异常（比如全是 -1）

# ============================================================
# Athena 连接
# ============================================================
class Athena_Connector:
    def __init__(self):
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
    """KDE：找两峰之间的密度谷底作为阈值"""
    data = data[np.isfinite(data)]
    if len(data) < 10:
        raise ValueError("数据太少，无法做 KDE")

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
        return _otsu_fallback(data)

    peaks.sort(key=lambda p: p[1], reverse=True)
    lo, hi = min(peaks[0][0], peaks[1][0]), max(peaks[0][0], peaks[1][0])

    valleys = []
    for i in range(1, len(density) - 1):
        if density[i] < density[i - 1] and density[i] < density[i + 1]:
            if lo <= grid_x[i] <= hi:
                valleys.append((grid_x[i], density[i]))

    if not valleys:
        return _otsu_fallback(data)

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
def clean_vibration(vibration: np.ndarray) -> tuple:
    is_valid = np.isfinite(vibration)
    return vibration[is_valid], is_valid

# 速度数据质量检查（新增）
# ============================================================
def check_speed_quality(speed: np.ndarray) -> dict:
    """
    检查速度数据是否可靠，返回质量报告。
    返回：
      - is_reliable: bool，速度数据是否足够可靠，可以用作验证基准
      - reason: str，不可靠的原因
      - stats: dict，各项统计数据
    """
    valid = np.isfinite(speed)& (speed >= 0)
    spd = speed[valid]
    total = len(spd)

    if total == 0:
        return {"is_reliable": False, "reason": "速度数据全部为 NaN/Inf", "stats": {}}

    unique_vals = np.unique(spd)
    n_unique = len(unique_vals)
    n_positive = int(np.sum(spd > SPEED_MIN_THRESHOLD))
    n_negative = int(np.sum(spd < 0))
    n_zero = int(np.sum((spd >= 0) & (spd <= SPEED_MIN_THRESHOLD)))
    ratio_positive = n_positive / total

    stats = {
        "total_valid": total,
        "n_unique": n_unique,
        "unique_vals_sample": unique_vals[:10].tolist(),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero_near": n_zero,
        "ratio_positive": ratio_positive,
        "min": float(spd.min()),
        "max": float(spd.max()),
        "mean": float(spd.mean()),
    }

    # 判断条件
    if n_unique < SPEED_UNIQUE_MIN:
        return {
            "is_reliable": False,
            "reason": f"速度唯一值只有 {n_unique} 个（< {SPEED_UNIQUE_MIN}），疑似传感器未接入或数据通道故障",
            "stats": stats,
        }
    if n_negative == total:
        return {
            "is_reliable": False,
            "reason": "速度全部为负值，数据无效",
            "stats": stats,
        }
    if ratio_positive < SPEED_VALID_RATIO:
        return {
            "is_reliable": False,
            "reason": f"正速度样本仅占 {ratio_positive*100:.2f}%（< {SPEED_VALID_RATIO*100}%），数据不足以验证",
            "stats": stats,
        }

    return {"is_reliable": True, "reason": "", "stats": stats}

# ============================================================
# 评估函数（速度做真值）—— 改进版
def evaluate_by_speed(vibration, speed, vib_threshold):
    """
    用速度作为真值评估振动阈值。
    关键改动：运行状态改为 speed > SPEED_MIN_THRESHOLD，而不是 speed != 0
    这样可以排除 -1 等无效负值的干扰。
    """
    valid = np.isfinite(speed) & np.isfinite(vibration)& (speed >= 0)
    vib_v, spd_v = vibration[valid], speed[valid]

    # 【核心改动】只有速度大于合理阈值才算"运行中"
    true_running = spd_v > SPEED_MIN_THRESHOLD
    pred_running = vib_v > vib_threshold

    tp = int(np.sum(pred_running & true_running))
    fp = int(np.sum(pred_running & ~true_running))
    fn = int(np.sum(~pred_running & true_running))
    tn = int(np.sum(~pred_running & ~true_running))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_valid": len(vib_v),
        "true_running_count": int(np.sum(true_running)),
        "true_stopped_count": int(np.sum(~true_running)),
    }

# ============================================================
# 评估函数（人工阈值做预测，速度做真值）—— 改进版
def evaluate_manual(vibration, speed, manual_threshold):
    valid = np.isfinite(speed) & np.isfinite(vibration)& (speed >= 0)
    vib_v, spd_v = vibration[valid], speed[valid]

    # 【核心改动】同样用 > SPEED_MIN_THRESHOLD 判断运行
    true_running = spd_v > SPEED_MIN_THRESHOLD
    pred_running = vib_v > manual_threshold

    tp = int(np.sum(pred_running & true_running))
    fp = int(np.sum(pred_running & ~true_running))
    fn = int(np.sum(~pred_running & true_running))
    tn = int(np.sum(~pred_running & ~true_running))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_valid": len(vib_v),
        "true_running_count": int(np.sum(true_running)),
        "true_stopped_count": int(np.sum(~true_running)),
    }

# ============================================================
# 生成图表（振动 + 速度 + 阈值标线）
# ============================================================
def create_chart(df, kde_threshold, manual_threshold=None):
    if df.empty:
        raise ValueError("查询结果为空，无法生成图表")

    fig = go.Figure()

    # 振动曲线（左轴）
    fig.add_trace(go.Scatter(
        x=df["event_time_shanghai"],
        y=df["gbvibforwardrms"],
        mode="lines+markers",
        name="振动 (gbvibforwardrms)",
        marker=dict(size=5),
        line=dict(width=2, color="blue"),
        hovertemplate="时间=%{x}<br>振动=%{y:.3f}<extra></extra>",
    ))
    # 速度曲线（右轴）
    fig.add_trace(go.Scatter(
        x=df["event_time_shanghai"],
        y=df["stepbandspeedleftavg"],
        mode="lines+markers",
        name="速度 (stepbandspeedleftavg)",
        marker=dict(size=5),
        line=dict(width=2, color="red"),
        yaxis="y2",
        hovertemplate="时间=%{x}<br>速度=%{y:.3f}<extra></extra>",
    ))
    # KDE 阈值线
    fig.add_hline(
        y=kde_threshold,
        line_dash="dash",
        line_color="green",
        line_width=2,
        annotation_text=f"KDE阈值={kde_threshold:.4f}",
        annotation_position="top left",
        annotation_font=dict(color="green", size=12),
    )

    # 人工阈值线
    if manual_threshold is not None:
        fig.add_hline(
            y=manual_threshold,
            line_dash="dash",
            line_color="orange",
            line_width=2,
            annotation_text=f"人工阈值={manual_threshold}",
            annotation_position="bottom left",
            annotation_font=dict(color="orange", size=12),
        )

    fig.update_layout(
        title=f"设备 {EQUIPMENT_NUMBERS} 振动与速度 ({START_DATE} ~ {END_DATE})",
        xaxis=dict(title="时间", tickangle=45),
        yaxis=dict(
            title="振动",
            title_font=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title="速度",
            title_font=dict(color="red"),
            tickfont=dict(color="red"),
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=600,
    )
    return fig

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    VIBRATION_COL = 'gbvibforwardrms'
    SPEED_COL = 'stepbandspeedleftavg'

    print(f"设备: {EQUIPMENT_NUMBERS}")
    print(f"日期: {START_DATE} ~ {END_DATE}（分界: {SPLIT_DATE}）")

    # 1. 查询数据
    print("\n正在从 Athena 查询数据……")
    eq_list = ", ".join(f"'{e}'" for e in EQUIPMENT_NUMBERS)
    sql = f"""
        SELECT equipmentnumber, gbvibforwardrms, stepbandspeedleftavg,
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

    print(f"查询到 {len(combined)} 条数据")

    # 去重
    dedup_cols = ['equipmentnumber', 'event_time_shanghai']
    combined = combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    # 2. 测试/验证集划分
    combined['date_str'] = combined['event_time_shanghai'].astype(str).str[:10]
    test_df = combined[combined['date_str'] <= SPLIT_DATE].copy()
    val_df = combined[combined['date_str'] > SPLIT_DATE].copy()

    # 3. 测试集：求 KDE 阈值
    vibration_test = test_df[VIBRATION_COL].values.astype(float)
    cleaned_test, _ = clean_vibration(vibration_test)
    threshold = kde_threshold(cleaned_test)

    # 4. 验证集准备
    vibration_val = val_df[VIBRATION_COL].values.astype(float)
    speed_val = val_df[SPEED_COL].values.astype(float)
    _, is_valid_val = clean_vibration(vibration_val)
    removed_val = len(vibration_val) - is_valid_val.sum()

    eval_mask = np.isfinite(vibration_val) & np.isfinite(speed_val) & (speed_val >= 0)
    vib_val_raw = vibration_val[eval_mask]
    spd_val_raw = speed_val[eval_mask]

    # 【新增】速度数据质量检查
    speed_quality = check_speed_quality(spd_val_raw)

    # 5. 输出结果
    print(f"\n{'='*60}")
    print(f"设备号: {EQUIPMENT_NUMBERS}")
    print(f"  测试集: {len(test_df)} 条 → 求阈值")
    print(f"  验证集: {len(val_df)} 条 → 验证阈值")
    print(f"{'='*60}")

    # 速度数据质量报告
    stats = speed_quality["stats"]
    print(f"\n速度数据质量检查:")
    if speed_quality["is_reliable"]:
        print(f"  ✅ 速度数据正常")
    else:
        print(f"  ⚠️  速度数据不可靠: {speed_quality['reason']}")
    print(f"\n验证集清洗: 移除 {removed_val} 个，保留 {is_valid_val.sum()} 个")
    print(f"\nKDE 振动阈值: {threshold:.4f}")

    # 速度验证（仅在速度数据可靠时输出）
    if speed_quality["is_reliable"]:
        m_speed = evaluate_by_speed(vib_val_raw, spd_val_raw, threshold)
        print(f"\n速度验证（验证集）:")
        print(f"  真实运行样本: {m_speed['true_running_count']} 条")
        print(f"  真实停机样本: {m_speed['true_stopped_count']} 条")
        print(f"  准确率 (Precision): {m_speed['precision']:.4f}")
        print(f"  覆盖率 (Recall):    {m_speed['recall']:.4f}")
        print(f"  F1-score:           {m_speed['f1_score']:.4f}")
        print(f"  TP={m_speed['tp']}, FP={m_speed['fp']}, FN={m_speed['fn']}, TN={m_speed['tn']}")

        if MANUAL_THRESHOLD is not None:
            m_manual = evaluate_manual(vib_val_raw, spd_val_raw, MANUAL_THRESHOLD)
            print(f"\n人工阈值验证（验证集）:")
            print(f"  人工阈值: {MANUAL_THRESHOLD}")
            print(f"  准确率 (Precision): {m_manual['precision']:.4f}")
            print(f"  覆盖率 (Recall):    {m_manual['recall']:.4f}")
            print(f"  F1-score:           {m_manual['f1_score']:.4f}")
            print(f"  TP={m_manual['tp']}, FP={m_manual['fp']}, FN={m_manual['fn']}, TN={m_manual['tn']}")
    else:
        print(f"\n速度验证（验证集）:")
        print(f"  ⚠️  已跳过 —— {speed_quality['reason']}")
        print(f"  建议: 检查速度传感器是否正常接入，或换用其他字段（如电流、运行状态位）验证")
        if MANUAL_THRESHOLD is not None:
            print(f"\n人工阈值验证:")
            print(f"  ⚠️  已跳过 —— 速度数据不可靠")

    # 6. 保存 CSV + 生成图表
    eq_tag = "_".join(EQUIPMENT_NUMBERS)
    output_dir = os.path.join(OUTPUT_DIR, eq_tag)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{eq_tag}.csv")
    combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nCSV 已保存: {csv_path}")

    print("\n正在生成图表……")
    fig = create_chart(combined, threshold, MANUAL_THRESHOLD)

    try:
        png_path = os.path.join(output_dir, f"{eq_tag}.png")
        fig.write_image(png_path, width=1200, height=600, scale=2)
        print(f"静态图片已保存: {png_path}")
    except Exception as e:
        print(f"[提示] 静态图片导出失败")
    fig.show()
