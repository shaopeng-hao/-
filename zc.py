"""
Athena 数据查询 + 固定阈值状态判断 + upperpitnoisepeak 真值对比验证 + CSV导出 + 图表生成
================================================================
功能：
  1. 从 Athena 查询振动数据（含 upperpitnoisepeak）
  2. 用固定阈值判定 gbvibforwardrms 状态（>阈值=运行1，否则=停梯0），作为“判定状态”
  3. 用固定阈值把 upperpitnoisepeak 转成 0/1，作为“真值状态”
  4. 按月对比验证：输出精确率、覆盖率、F1、混淆矩阵
  5. 生成图表：振动曲线（左轴）+ 真值状态线 + 判定状态线（右轴）+ 阈值标线
  6. 保存 CSV（新增 gt_status / pred_status 两列）和 PNG
  7. 额外保存按月汇总 CSV
"""

import os
import warnings

import boto3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyathena

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

EQUIPMENT_NUMBERS = ['30227409']
START_DATE = '2026-07-01'
END_DATE = '2026-09-20'

STATUS_THRESHOLD = 4.0
NOISE_STATUS_THRESHOLD = -45
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

VIBRATION_COL = 'gbvibforwardrms'
NOISE_COL = 'upperpitnoisepeak'


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


def clean_vibration(vibration: np.ndarray) -> tuple:
    is_valid = np.isfinite(vibration)
    return vibration[is_valid], is_valid


def derive_status(df: pd.DataFrame, vib_col: str, noise_col: str) -> pd.DataFrame:
    vib = df[vib_col].astype(float)
    noise = df[noise_col].astype(float)

    df["pred_status"] = np.where(
        vib.notna(),
        (vib > STATUS_THRESHOLD).astype(float),
        np.nan
    )
    df["gt_status"] = np.where(
        noise.notna(),
        (noise > NOISE_STATUS_THRESHOLD).astype(float),
        np.nan
    )
    return df


def evaluate_against_gt(df: pd.DataFrame) -> dict:
    valid = df["pred_status"].notna() & df["gt_status"].notna()
    pred = df.loc[valid, "pred_status"].astype(int).values
    gt = df.loc[valid, "gt_status"].astype(int).values

    tp = int(((pred == 1) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())

    total = len(gt)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")

    return {
        "total": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision * 100,   # 变成百分数
        "recall": recall * 100,        # 变成百分数
        "f1": f1 * 100,                # 变成百分数
    }



def summarize_by_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_time_shanghai"] = pd.to_datetime(df["event_time_shanghai"], errors="coerce")
    df["month"] = df["event_time_shanghai"].dt.strftime("%Y-%m")

    df["month"] = df["event_time_shanghai"].dt.strftime("%Y-%m") 

    rows = []
    for month_label, g in df.groupby("month", sort=True):
        g = g.copy()
        g = derive_status(g, VIBRATION_COL, NOISE_COL)
        metrics = evaluate_against_gt(g)

        if len(g) == 0:
            continue

        equipment_value = g["equipmentnumber"].dropna()
        eq = equipment_value.iloc[0] if not equipment_value.empty else ""

        rows.append({
            "equipmentnumber": g["equipmentnumber"].iloc[0],
            "month": month_label,
            "vibration_threshold": STATUS_THRESHOLD,
            "noise_threshold": NOISE_STATUS_THRESHOLD,
            "precision": metrics["precision"],   # 已经是百分数
            "recall": metrics["recall"],        # 已经是百分数
            "f1": metrics["f1"],                # 已经是百分数
            "valid_total": metrics["total"],

        })

    return pd.DataFrame(rows)


def create_noise_chart(df: pd.DataFrame, noise_threshold: float):
    if df.empty:
        raise ValueError("查询结果为空，无法生成图表")

    fig = go.Figure()

    noise_values = df[NOISE_COL].astype(float).values
    fig.add_trace(go.Scatter(
        x=df["event_time_shanghai"],
        y=noise_values,
        mode="lines+markers",
        name="upperpitnoisepeak",
        marker=dict(size=5),
        line=dict(width=3, color="purple"),
        hovertemplate="时间=%{x}<br>upperpitnoisepeak=%{y:.3f}<extra></extra>",
    ))

    noise_status = np.where(np.isfinite(noise_values), (noise_values > noise_threshold).astype(int), np.nan)
    fig.add_trace(go.Scatter(
        x=df["event_time_shanghai"],
        y=noise_status,
        mode="lines",
        name=f"状态 (upperpitnoisepeak > {noise_threshold})",
        line=dict(width=2, color="#ff6b35"),
        yaxis="y2",
        hovertemplate="时间=%{x}<br>状态=%{y}<extra></extra>",
        connectgaps=False,
    ))

    fig.add_hline(
        y=noise_threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"阈值={noise_threshold}",
        annotation_position="top left",
        annotation_font=dict(color="red", size=12),
    )

    fig.update_layout(
        title=f"设备 {EQUIPMENT_NUMBERS} upperpitnoisepeak 与状态 ({START_DATE} ~ {END_DATE})",
        xaxis=dict(title="时间", tickangle=45),
        yaxis=dict(title="upperpitnoisepeak", title_font=dict(color="purple"), tickfont=dict(color="purple")),
        yaxis2=dict(
            title="状态 (0/1)",
            title_font=dict(color="#ff6b35"),
            tickfont=dict(color="#ff6b35"),
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


def create_chart(df, vib_threshold):
    if df.empty:
        raise ValueError("查询结果为空，无法生成图表")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["event_time_shanghai"],
        y=df["gbvibforwardrms"],
        mode="lines+markers",
        name="振动 (gbvibforwardrms)",
        marker=dict(size=5),
        line=dict(width=3, color="blue"),
        hovertemplate="时间=%{x}<br>振动=%{y:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df["event_time_shanghai"],
        y=df["gt_status"],
        mode="lines",
        name=f"真值状态 (upperpitnoisepeak>{NOISE_STATUS_THRESHOLD}dB)",
        line=dict(width=2, color="green"),
        yaxis="y2",
        hovertemplate="时间=%{x}<br>真值状态=%{y}<extra></extra>",
        connectgaps=False,
    ))

    pred = df["pred_status"].values
    pred_offset = np.where(np.isfinite(pred), 0.05 + 0.9 * pred, np.nan)
    fig.add_trace(go.Scatter(
        x=df["event_time_shanghai"],
        y=pred_offset,
        mode="lines",
        name=f"判定状态 (gbvibforwardrms>{vib_threshold})",
        line=dict(width=2, color="#ff8c42", dash="dot"),
        yaxis="y2",
        customdata=df["pred_status"],
        hovertemplate="时间=%{x}<br>判定状态=%{customdata}<extra></extra>",
        connectgaps=False,
    ))

    fig.add_hline(
        y=vib_threshold,
        line_dash="dash",
        line_color="#1f9d55",
        line_width=2,
        annotation_text=f"振动阈值={vib_threshold:.4f}",
        annotation_position="top left",
        annotation_font=dict(color="#1f9d55", size=12),
    )

    fig.update_layout(
        title=f"设备 {EQUIPMENT_NUMBERS} 振动判定 vs upperpitnoisepeak 真值 ({START_DATE} ~ {END_DATE})",
        xaxis=dict(title="时间", tickangle=45),
        yaxis=dict(title="振动", title_font=dict(color="blue"), tickfont=dict(color="blue")),
        yaxis2=dict(
            title="状态 (0/1)",
            title_font=dict(color="#ff8c42"),
            tickfont=dict(color="#ff8c42"),
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


if __name__ == "__main__":
    eq_list = ", ".join(f"'{e}'" for e in EQUIPMENT_NUMBERS)
    sql = f"""
        SELECT equipmentnumber, gbvibforwardrms, upperpitnoisepeak,
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

    combined = combined.drop_duplicates(subset=['equipmentnumber', 'event_time_shanghai']).reset_index(drop=True)
    combined = derive_status(combined, VIBRATION_COL, NOISE_COL)

    eq_tag = "_".join(EQUIPMENT_NUMBERS)
    output_dir = os.path.join(OUTPUT_DIR, eq_tag)
    os.makedirs(output_dir, exist_ok=True)

    monthly_df = summarize_by_month(combined)

    # 先把百分值转成 "xx.xx%" 字符串，再写入 CSV
    for col in ["precision", "recall", "f1"]:
         monthly_df[col] = monthly_df[col].map(lambda x: f"{x:.2f}%" if pd.notna(x) else x)

         monthly_path = os.path.join(output_dir, f"{eq_tag}_monthly_metrics_{START_DATE}_{END_DATE}.csv")
         monthly_df.to_csv(monthly_path, index=False, encoding='utf-8-sig')

    print("\n按月指标:")
    print(monthly_df[[
        "equipmentnumber",
        "month",
        "vibration_threshold",
        "noise_threshold",
        "precision",
        "recall",
        "f1",
        "valid_total"
    ]].to_string(index=False))


    file_tag = f"{eq_tag}_{START_DATE}_{END_DATE}"
    csv_path = os.path.join(output_dir, f"{file_tag}.csv")
    combined.to_csv(csv_path, index=False, encoding='utf-8-sig')


    fig = create_chart(combined, STATUS_THRESHOLD)
    png_path = os.path.join(output_dir, f"{file_tag}_compare.png")
    fig.write_image(png_path, width=1200, height=600, scale=2)


    noise_fig = create_noise_chart(combined, NOISE_STATUS_THRESHOLD)
    noise_png_path = os.path.join(output_dir, f"{file_tag}_noise.png")
    noise_fig.write_image(noise_png_path, width=1200, height=600, scale=2)
    print("完成")