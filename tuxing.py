from pathlib import Path
import warnings

import boto3
import pandas as pd
import plotly.graph_objects as go
import pyathena

# 屏蔽 pyathena 直连触发的 SQLAlchemy 警告（无害，不影响查询）
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / "test1.csv"
HTML_PATH = OUTPUT_DIR / "tuxing.html"
IMAGE_PATH = OUTPUT_DIR / "tuxing.png"


class AthenaConnector:
    def __init__(self):
        self.conn = pyathena.connect(
            s3_staging_dir="s3://athena-query-prod-zhen/",
            session=boto3.Session(profile_name="AWSPowerUserAccess-579289528406"),
        )

    def query(self, sql):
        return pd.read_sql(sql, self.conn)

    def close(self):
        self.conn.close()


def query_athena_data():
    sql = """
        SELECT equipmentnumber, gbvibforwardrms, motorcurrent1avg, stepbandspeedleftavg,
               modeset, operationstatus,
               (FROM_ISO8601_TIMESTAMP(timestamp) AT TIME ZONE 'Asia/Shanghai') AS event_time_shanghai
        FROM data_cleansed."anyescalator"
        WHERE equipmentnumber IN ('45024583')
          AND eventdate BETWEEN '2026-07-21' AND '2026-07-26'
        ORDER BY timestamp
    """

    connector = AthenaConnector()
    try:
        return connector.query(sql)
    finally:
        connector.close()


def create_chart(df):
    if df.empty:
        raise ValueError("Athena 查询结果为空，无法生成图表。请检查设备编号和日期范围。")

    required_columns = {
        "event_time_shanghai",
        "gbvibforwardrms",
        "stepbandspeedleftavg",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"查询结果缺少绘图字段: {', '.join(sorted(missing_columns))}")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["event_time_shanghai"],
            y=df["gbvibforwardrms"],
            mode="lines+markers",
            name="gbvibforwardrms",
            marker=dict(size=6),
            line=dict(width=2),
            hovertemplate="time=%{x}<br>gbvibforwardrms=%{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["event_time_shanghai"],
            y=df["stepbandspeedleftavg"],
            mode="lines+markers",
            name="stepbandspeedleftavg",
            marker=dict(size=6),
            line=dict(width=2),
            yaxis="y2",
            hovertemplate="time=%{x}<br>stepbandspeedleftavg=%{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="gbvibforwardrms and stepbandspeedleftavg vs event_time_shanghai",
        xaxis=dict(title="event_time_shanghai", tickangle=45),
        yaxis=dict(
            title="gbvibforwardrms",
            title_font=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title="stepbandspeedleftavg",
            title_font=dict(color="red"),
            tickfont=dict(color="red"),
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def main():
    print("正在从 Athena 查询数据……")
    df = query_athena_data()

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"CSV 已保存: {CSV_PATH}")

    fig = create_chart(df)
    fig.write_html(HTML_PATH, include_plotlyjs="cdn")
    print(f"交互式图表已保存: {HTML_PATH}")

    # Plotly 导出 PNG 需要安装 kaleido：pip install kaleido
    fig.write_image(IMAGE_PATH, width=1200, height=600, scale=2)
    print(f"静态图片已保存: {IMAGE_PATH}")

    fig.show()


if __name__ == "__main__":
    main()
