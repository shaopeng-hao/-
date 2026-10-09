"""
Athena 数据查询 + KDE阈值 + CSV导出 + 图表生成
================================================
从 Athena 查询扶梯数据，用 KDE 求振动阈值，速度做验证，
保存 CSV 和图表（含人工阈值、KDE阈值标线）
运行：python tu.py
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


# ============================================================
# 配置（改成你要查询的设备和日期）
# ============================================================
EQUIPMENT_NUMBERS = ['45024583']
START_DATE = '2026-08-20'
END_DATE = '2026-08-24'
SPLIT_DATE = '2026-08-22'          # <= 求阈值，> 验证
MANUAL_THRESHOLD = 1.5             # 人工阈值，设为 None 则不验证
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

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
# 数据清洗（只去 NaN / Inf）
# ============================================================
def clean_vibration(vibration: np.ndarray) -> tuple:
    is_valid = np.isfinite(vibration)
    return vibration[is_valid], is_valid


# ============================================================
# 评估函数（速度做真值）
# ============================================================
def evaluate_by_speed(vibration, speed, vib_threshold):
    valid = np.isfinite(speed) & np.isfinite(vibration)
    vib_v, spd_v = vibration[valid], speed[valid]
    true_running = spd_v != 0
    pred_running = vib_v > vib_threshold

    tp = int(np.sum(pred_running & true_running))
    fp = int(np.sum(pred_running & ~true_running))
    fn = int(np.sum(~pred_running & true_running))
    tn = int(np.sum(~pred_running & ~true_running))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}


# ============================================================
# 评估函数（人工阈值做预测，速度做真值）
# ============================================================
def evaluate_manual(vibration, speed, manual_threshold):
    valid = np.isfinite(speed) & np.isfinite(vibration)
    vib_v, spd_v = vibration[valid], speed[valid]
    true_running = spd_v != 0
    pred_running = vib_v > manual_threshold

    tp = int(np.sum(pred_running & true_running))
    fp = int(np.sum(pred_running & ~true_running))
    fn = int(np.sum(~pred_running & true_running))
    tn = int(np.sum(~pred_running & ~true_running))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}


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
        SELECT equipmentnumber, gbvibforwardrms, motorcurrent1avg, stepbandspeedleftavg,
               modeset, operationstatus,
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

    # 4. 验证集：验证
    vibration_val = val_df[VIBRATION_COL].values.astype(float)
    speed_val = val_df[SPEED_COL].values.astype(float)
    _, is_valid_val = clean_vibration(vibration_val)
    removed_val = len(vibration_val) - is_valid_val.sum()

    eval_mask = np.isfinite(vibration_val) & np.isfinite(speed_val)
    vib_val_raw = vibration_val[eval_mask]
    spd_val_raw = speed_val[eval_mask]

    m_speed = evaluate_by_speed(vib_val_raw, spd_val_raw, threshold)
    m_manual = evaluate_manual(vib_val_raw, spd_val_raw, MANUAL_THRESHOLD) if MANUAL_THRESHOLD is not None else None

    # 5. 输出结果
    print(f"\n{'='*60}")
    print(f"设备号: {EQUIPMENT_NUMBERS}")
    print(f"  测试集: {len(test_df)} 条 → 求阈值")
    print(f"  验证集: {len(val_df)} 条 → 验证阈值")
    print(f"{'='*60}")

    valid_spd = speed_val[np.isfinite(speed_val)]
    print(f"\n速度数据: 可用")
    print(f"  速度 != 0（运行）: {int(np.sum(valid_spd != 0))} 条")
    print(f"  速度 == 0（停机）: {int(np.sum(valid_spd == 0))} 条")

    print(f"\n验证集清洗: 移除 {removed_val} 个，保留 {is_valid_val.sum()} 个")
    print(f"\nKDE 振动阈值: {threshold:.4f}")

    print(f"\n速度验证（验证集）:")
    print(f"  准确率 (Precision): {m_speed['precision']:.4f}")
    print(f"  覆盖率 (Recall):    {m_speed['recall']:.4f}")
    print(f"  F1-score:           {m_speed['f1_score']:.4f}")

    if m_manual is not None:
        print(f"\n人工阈值验证（验证集）:")
        print(f"  人工阈值: {MANUAL_THRESHOLD}")
        print(f"  准确率 (Precision): {m_manual['precision']:.4f}")
        print(f"  覆盖率 (Recall):    {m_manual['recall']:.4f}")
        print(f"  F1-score:           {m_manual['f1_score']:.4f}")

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
        print(f"[提示] 静态图片导出失败，需要安装 kaleido: pip install kaleido")
        print(f"  错误: {e}")

    fig.show()
