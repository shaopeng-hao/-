"""
Athena 数据查询 + 固定阈值状态判断 + CSV导出 + 图表生成
================================================
功能：
  1. 从 Athena 查询振动数据
  2. 清洗数据（去除 NaN）
  3. 用 固定阈值  和status_threshold判定状态
  4. 生成图表：振动曲线（左轴）+ 0/1状态线（右轴）+ 阈值标线
  5. 保存 CSV 和 PNG

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
EQUIPMENT_NUMBERS = ['45896520']
START_DATE = '2026-07-01'
END_DATE = '2026-07-24'

#START_DATE = '2026-09-01'
#END_DATE = '2026-09-03'

STATUS_THRESHOLD = 4# 振动值大于该值时，状态为 1
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
# 数据清洗
# ============================================================
def clean_vibration(vibration: np.ndarray) -> tuple:
    is_valid = np.isfinite(vibration)
    return vibration[is_valid], is_valid


# ============================================================
# 生成图表（振动 + 0/1状态线 + 阈值标线）
# ============================================================
def create_chart(df, kde_threshold):
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
        line=dict(width=3, color="darkblue"),
        hovertemplate="时间=%{x}<br>振动=%{y:.3f}<extra></extra>",
    ))

    # 0/1 状态线（右轴）：大于阈值=1，小于等于阈值=0
    vib_values = df["gbvibforwardrms"].values
    status_01 = np.where(vib_values > kde_threshold, 1, 0)
    fig.add_trace(go.Scatter(
        x=df["event_time_shanghai"],
        y=status_01,
        mode="lines",
        name="状态 (0=低于阈值, 1=高于阈值)",
        line=dict(width=1, color="lightsalmon"),
        yaxis="y2",
        hovertemplate="时间=%{x}<br>状态=%{y}<extra></extra>",
    ))

    # KDE 阈值线
    fig.add_hline(
        y=kde_threshold,
        line_dash="dash",
        line_color="green",
        line_width=2,
        annotation_text=f"阈值={kde_threshold:.4f}",
        annotation_position="top left",
        annotation_font=dict(color="green", size=12),
    )

    fig.update_layout(
        title=f"设备 {EQUIPMENT_NUMBERS} 振动与状态 ({START_DATE} ~ {END_DATE})",
        xaxis=dict(title="时间", tickangle=45),
        yaxis=dict(
            title="振动",
            title_font=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title="状态 (0/1)",
            title_font=dict(color="lightsalmon"),
            tickfont=dict(color="lightsalmon"),
            overlaying="y",
            side="right",
            range=[-0.2, 1.2],
            dtick=1,
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

    print(f"设备: {EQUIPMENT_NUMBERS}")
    print(f"日期: {START_DATE} ~ {END_DATE}")

    # 1. 查询数据
    print("\n正在从 Athena 查询数据……")
    eq_list = ", ".join(f"'{e}'" for e in EQUIPMENT_NUMBERS)
    sql = f"""
        SELECT equipmentnumber, gbvibforwardrms, motorcurrent1avg, stepbandspeedleftavg,passengercount,
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

    # 2. 准备振动数据
    vibration_all = combined[VIBRATION_COL].values.astype(float)
    cleaned_vib, is_valid = clean_vibration(vibration_all)
    removed = len(vibration_all) - len(cleaned_vib)

    # 3. KDE 求阈值
    threshold = STATUS_THRESHOLD

    # 4. 输出
    print(f"\n{'='*60}")
    print(f"设备号: {EQUIPMENT_NUMBERS}")
    print(f"  数据量: {len(combined)} 条")
    print(f"  剔除无效(振动NaN): {removed} 条")
    print(f"  有效数据: {len(cleaned_vib)} 条")
    print(f"{'='*60}")
    print(f"\nKDE 振动阈值: {threshold:.4f}")

    # 5. 保存 CSV + 生成图表
    eq_tag = "_".join(EQUIPMENT_NUMBERS)
    output_dir = os.path.join(OUTPUT_DIR, eq_tag)
    os.makedirs(output_dir, exist_ok=True)

    file_tag = f"{eq_tag}_{START_DATE}_{END_DATE}"

    csv_path = os.path.join(output_dir, f"{file_tag}.csv")
    combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nCSV 已保存: {csv_path}")

    print("\n正在生成图表……")
    fig = create_chart(combined, threshold)

    try:
        png_path = os.path.join(output_dir, f"{file_tag}.png")
        fig.write_image(png_path, width=1200, height=600, scale=2)
        print(f"静态图片已保存: {png_path}")
    except Exception as e:
        print(f"[提示] 静态图片导出失败，需要安装 kaleido: pip install kaleido")
        print(f"  错误: {e}")

    fig.show()
