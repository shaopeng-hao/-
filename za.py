"""
Athena 数据查询  + CSV导出 + 图表生成
================================================
功能：
  1. 从 Athena 查询振动数据
  4. 生成图表：振动曲线（左轴）+ 0/1状态线（右轴）+ 阈值标线+passenger_count
  5. 保存 PNG

import os
import warnings
from pathlib import Path
import boto3
import pandas as pd
import plotly.graph_objects as go
import pyathena
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

# 配置
EQUIPMENT_NUMBERS = ["45024582"]
START_DATE = "2026-09-10"
END_DATE = "2026-09-12"

STATUS_THRESHOLD = 4.0
OUTPUT_DIR = Path(r"C:\Users\k64183565\OneDrive - KONE Corporation\桌面\数据")
SAVE_CSV = False
SAVE_PNG = True


class AthenaConnector:
    def __init__(self):
        self.conn = pyathena.connect(
            s3_staging_dir="s3://athena-query-prod-zhen/",
            session=boto3.Session(
                profile_name="AWSPowerUserAccess-579289528406"
            ),
        )

    def query(self, sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, self.conn)

    def close(self):
        self.conn.close()


def count_cn_events_by_month(
    df: pd.DataFrame,
    threshold: float,
) -> dict[str, dict[str, int]]:
    """统计每月、每台设备：振动低于阈值且乘客数大于 0 的数据条数。"""
    data = df.copy()

    data["event_time_shanghai"] = pd.to_datetime(
        data["event_time_shanghai"],
        errors="coerce",
    )
    data["gbvibforwardrms"] = pd.to_numeric(
        data["gbvibforwardrms"],
        errors="coerce",
    )
    data["passengercount"] = pd.to_numeric(
        data["passengercount"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "event_time_shanghai",
            "gbvibforwardrms",
            "passengercount",
        ]
    )

    data["query_month"] = data["event_time_shanghai"].dt.strftime("%Y-%m")

    result = {}

    for (month, equipment), group in data.groupby(
        ["query_month", "equipmentnumber"]
    ):
        cn_count = (
            group["gbvibforwardrms"].lt(threshold)
            & group["passengercount"].gt(0)
        ).sum()

        result.setdefault(month, {})[str(equipment)] = int(cn_count)

    return result


def create_chart(df: pd.DataFrame, threshold: float):
    if df.empty:
        raise ValueError("查询结果为空，无法生成图表")

    vibration = pd.to_numeric(
        df["gbvibforwardrms"],
        errors="coerce",
    )

    passenger_count = pd.to_numeric(
        df["passengercount"],
        errors="coerce",
    )

    status = vibration.ge(threshold).astype(int)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["event_time_shanghai"],
            y=vibration,
            mode="lines+markers",
            name="振动",
            marker=dict(size=5),
            line=dict(width=3, color="darkblue"),
            hovertemplate="时间=%{x}<br>振动=%{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["event_time_shanghai"],
            y=passenger_count,
            mode="lines+markers",
            name="乘客数",
            marker=dict(
                size=3,
                color="rgba(255, 165, 0, 0.45)",
            ),
            line=dict(
                width=1,
                color="rgba(255, 165, 0, 0.45)",
            ),
            yaxis="y2",
            hovertemplate="时间=%{x}<br>乘客数=%{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["event_time_shanghai"],
            y=status,
            mode="lines",
            name="状态<br>(0=低于阈值，1=达到阈值)",
            line=dict(width=1.5, color="lightsalmon"),
            yaxis="y3",
            hovertemplate="时间=%{x}<br>状态=%{y}<extra></extra>",
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"阈值={threshold:.4f}",
        annotation_position="top left",
    )

    fig.update_layout(
        title=f"设备 {EQUIPMENT_NUMBERS} 振动、乘客数与状态 "
              f"({START_DATE} ~ {END_DATE})",
        xaxis=dict(title="时间", tickangle=45),
        yaxis=dict(title="振动"),
        yaxis2=dict(
            title="乘客数",
            overlaying="y",
            side="right",
        ),
        yaxis3=dict(
            overlaying="y",
            side="left",
            range=[-0.1, 1.1],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        margin=dict(l=60, r=45, t=80, b=100),
        hovermode="x unified",
        template="plotly_white",
        height=600,
        autosize=True,
    )

    return fig


def main():
    print(f"设备: {EQUIPMENT_NUMBERS}")
    print(f"日期: {START_DATE} ~ {END_DATE}")
    print("\n正在从 Athena 查询数据……")

    equipment_sql = ", ".join(
        f"'{equipment}'" for equipment in EQUIPMENT_NUMBERS
    )

    sql = f"""
        SELECT
            equipmentnumber,
            gbvibforwardrms,
            passengercount,
            (
                FROM_ISO8601_TIMESTAMP(timestamp)
                AT TIME ZONE 'Asia/Shanghai'
            ) AS event_time_shanghai
        FROM data_cleansed."anyescalator"
        WHERE equipmentnumber IN ({equipment_sql})
          AND eventdate BETWEEN '{START_DATE}' AND '{END_DATE}'
        ORDER BY timestamp
    """

    connector = AthenaConnector()

    try:
        combined = connector.query(sql)
    finally:
        connector.close()

    print(f"查询到 {len(combined)} 条数据")

    combined = combined.drop_duplicates(
        subset=["equipmentnumber", "event_time_shanghai"]
    ).reset_index(drop=True)

    monthly_cn = count_cn_events_by_month(
        combined,
        STATUS_THRESHOLD,
    )

    print("\n按月 CN 统计结果：")

    total_cn = 0

    for month in sorted(monthly_cn):
        month_total = sum(monthly_cn[month].values())
        total_cn += month_total

        print(f"\n{month}：")

        for equipment, count in monthly_cn[month].items():
            print(f"  设备 {equipment}: {count} 次")

        print(f"  {month} CN 总次数: {month_total} 次")

    print(f"\n全部月份 CN 总次数: {total_cn} 次")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    equipment_tag = "_".join(EQUIPMENT_NUMBERS)
    date_tag = f"{START_DATE}至{END_DATE}"

    if SAVE_CSV:
        csv_path = OUTPUT_DIR / f"{equipment_tag}_{date_tag}.csv"
        combined.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"\nCSV 已保存: {csv_path}")

    if SAVE_PNG:
        print("\n正在生成图表……")

        fig = create_chart(combined, STATUS_THRESHOLD)
        png_path = OUTPUT_DIR / f"{equipment_tag}_{date_tag}.png"

        try:
            fig.write_image(
                str(png_path),
                width=1200,
                height=600,
                scale=2,
            )
            print(f"图片已保存: {png_path}")
        except Exception as error:
            print(f"图片导出失败: {error}")

        fig.show(
            config={
                "responsive": True,
                "displaylogo": False,
            }
        )


if __name__ == "__main__":
    main()
