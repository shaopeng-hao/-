"""统计指定设备的 CN 数据发送条数及收到数据的天数。

数据来源：Athena 的 data_cleansed.analytics_enriched_service_needs。
CN 数据定义为 service_need_code 以 ``CN`` 开头的记录；接收日期使用 event_date。
"""
import os
import time
import boto3
import pandas as pd

PROFILE = "AWSPowerUserAccess-579289528406"
REGION = "cn-north-1"
DATABASE = "data_cleansed"
TABLE = "enriched_service_needs"
WORKGROUP = "primary"
S3_STAGING_DIR = "s3://analytics-callout-2025"

EQUIPMENT_NUMBER = "45024582"
START_DATE = "2026-09-01"
END_DATE = "2026-09-20"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Athena 查询
# ============================================================
def run_athena_query(athena_client, sql: str) -> pd.DataFrame:
    """执行 Athena 查询并将结果转换为 DataFrame。"""
    response = athena_client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={
            "Database": DATABASE,
            "Catalog": "AwsDataCatalog",
        },
        ResultConfiguration={"OutputLocation": S3_STAGING_DIR},
        WorkGroup=WORKGROUP,
    )
    query_execution_id = response["QueryExecutionId"]

    while True:
        execution = athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )
        status = execution["QueryExecution"]["Status"]
        state = status["State"]
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            break
        time.sleep(1)

    if state != "SUCCEEDED":
        reason = status.get("StateChangeReason", "无详细错误信息")
        raise RuntimeError(f"Athena 查询失败（{state}）：{reason}")

    rows = []
    columns = []
    paginator = athena_client.get_paginator("get_query_results")
    for page_number, page in enumerate(
        paginator.paginate(QueryExecutionId=query_execution_id)
    ):
        result_set = page["ResultSet"]
        if page_number == 0:
            columns = [
                column["Label"]
                for column in result_set["ResultSetMetadata"]["ColumnInfo"]
            ]
            data_rows = result_set["Rows"][1:]  # 第一行是列名
        else:
            data_rows = result_set["Rows"]

        for row in data_rows:
            values = [cell.get("VarCharValue") for cell in row["Data"]]
            rows.append(values + [None] * (len(columns) - len(values)))

    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    print(f"设备号：{EQUIPMENT_NUMBER}")
    print(f"统计日期：{START_DATE} ~ {END_DATE}")
    print("CN 规则：service_need_code 以 CN 开头")
    print("\n正在从 Athena 查询 CN 数据……")
    # event_date 是表的日期字段；保留其过滤条件可避免扫描无关日期分区。
    daily_sql = f"""
        SELECT
            event_date,
            message_type,
            COUNT(*) AS record_count
        FROM {TABLE}
        WHERE CAST(eq AS VARCHAR) = '{EQUIPMENT_NUMBER}'
          AND service_need_code LIKE 'CN%'
          AND message_type IN (
              'ServiceNeed',
              'ServiceNeedCancellation'
          )
          AND event_date BETWEEN '{START_DATE}' AND '{END_DATE}'
        GROUP BY event_date, message_type
        ORDER BY event_date, message_type
    """

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    athena = session.client("athena")
    daily_df = run_athena_query(athena, daily_sql)

    # 查询并保存原始明细数据
    raw_sql = f"""
        SELECT
            eq,
            message_type,
            service_need_code,
            event_time
        FROM {TABLE}
        WHERE CAST(eq AS VARCHAR) = '{EQUIPMENT_NUMBER}'
          AND service_need_code LIKE 'CN%'
          AND message_type IN (
              'ServiceNeed',
              'ServiceNeedCancellation'
          )
          AND event_date BETWEEN '{START_DATE}' AND '{END_DATE}'
        ORDER BY event_time ASC
    """

    raw_df = run_athena_query(athena, raw_sql)

    output_dir = os.path.join(OUTPUT_DIR, EQUIPMENT_NUMBER)
    os.makedirs(output_dir, exist_ok=True)

    date_tag = f"{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}"

    raw_output_path = os.path.join(
        output_dir,
        f"{EQUIPMENT_NUMBER}_cn_raw_{date_tag}.csv",
    )

    raw_df.to_csv(
        raw_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"读取到的明细数据已保存：{raw_output_path}")

    if daily_df.empty:
        total_serviceneed = 0
        total_cancellation = 0
    else:
        daily_df["record_count"] = pd.to_numeric(
            daily_df["record_count"],
            errors="coerce",
        ).fillna(0).astype(int)

        total_serviceneed = int(
            daily_df.loc[
                daily_df["message_type"] == "ServiceNeed",
                "record_count",
            ].sum()
        )

        total_cancellation = int(
            daily_df.loc[
                daily_df["message_type"] == "ServiceNeedCancellation",
                "record_count",
            ].sum()
        )

    received_days = pd.to_datetime(raw_df["event_time"]).dt.date.nunique() if not raw_df.empty else 0

    print(f"\n{'=' * 48}")
    print("CN 数据统计结果")
    print(f"设备号：{EQUIPMENT_NUMBER}")
    print(f"日期范围：{START_DATE} ~ {END_DATE}")
    print(f"CN 数据发送条数：{total_serviceneed} 条")
    print(
        "ServiceNeedCancellation 数量："
        f"{total_cancellation} 条"
    )
    print(f"有数据的天数：{received_days} 天")

    output_dir = os.path.join(OUTPUT_DIR, EQUIPMENT_NUMBER)
    os.makedirs(output_dir, exist_ok=True)
    date_tag = f"{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}"
    output_path = os.path.join(
        output_dir, f"{EQUIPMENT_NUMBER}_cn_daily_{date_tag}.csv"
    )
    daily_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 在统计表末尾增加汇总行
    summary_row = pd.DataFrame(
        [{
            "event_date": "汇总",
            "message_type": "总天数 / 总CN条数",
            "record_count": (
                f"{received_days} / {total_serviceneed}"
            ),
        }]
    )

    output_df = pd.concat(
        [daily_df, summary_row],
        ignore_index=True,
    )

    output_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n每日统计已导出：{output_path}")

if __name__ == "__main__":
    main()
