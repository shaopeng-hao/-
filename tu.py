"""
Athena 数据查询 + CSV 导出 + 图表生成
=====================================
从 AWS Athena 查询扶梯数据，保存 CSV，并生成静态图片
依赖：numpy, pandas, pyathena, boto3, plotly, kaleido
安装：pip install numpy pandas pyathena boto3 plotly kaleido

运行：python tu.py
"""

import os
import warnings

import boto3
import pandas as pd
import plotly.graph_objects as go
import pyathena

# 屏蔽 pyathena 直连触发的 SQLAlchemy 警告（无害，不影响查询）
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")


# ============================================================
# 配置（改成你要查询的设备和日期）
# ============================================================
EQUIPMENT_NUMBERS = ['30536165']
START_DATE = '2026-07-20'
END_DATE = '2026-07-26'
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
# 查询数据
# ============================================================
def query_data():
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
        df = conn.query(sql)
    finally:
        conn.close()
    return df


# ============================================================
# 生成图表
# ============================================================
def create_chart(df):
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
    print(f"设备: {EQUIPMENT_NUMBERS}")
    print(f"日期: {START_DATE} ~ {END_DATE}")

    # 1. 查询数据
    print("\n正在从 Athena 查询数据……")
    df = query_data()
    print(f"查询到 {len(df)} 条数据")
    print(df.head())

    # 2. 创建设备号文件夹，CSV 和 PNG 都放进去
    eq_tag = "_".join(EQUIPMENT_NUMBERS)
    output_dir = os.path.join(OUTPUT_DIR, eq_tag)
    os.makedirs(output_dir, exist_ok=True)

    # 3. 保存 CSV
    csv_path = os.path.join(output_dir, f"{eq_tag}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nCSV 已保存: {csv_path}")

    # 4. 生成图表并保存 PNG
    print("\n正在生成图表……")
    fig = create_chart(df)

    try:
        png_path = os.path.join(output_dir, f"{eq_tag}.png")
        fig.write_image(png_path, width=1200, height=600, scale=2)
        print(f"静态图片已保存: {png_path}")
    except Exception as e:
        print(f"  错误: {e}")

    # 5. 显示图表
    fig.show()
