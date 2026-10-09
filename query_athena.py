import time
import boto3

PROFILE = "AWSPowerUserAccess-579289528406"
REGION = "cn-north-1"
DATABASE = "data_cleansed"
WORKGROUP = "primary"

SQL = """
SELECT *
FROM analytics_enriched_service_needs 
LIMIT 10
"""

session = boto3.Session(profile_name=PROFILE, region_name=REGION)
athena = session.client("athena")

response = athena.start_query_execution(
    QueryString=SQL,
    QueryExecutionContext={
        "Database": DATABASE,
        "Catalog": "AwsDataCatalog"
    },
    ResultConfiguration={
        "OutputLocation": "s3://analytics-callout-2025"
    },
    WorkGroup=WORKGROUP
)

query_execution_id = response["QueryExecutionId"]
print("QueryExecutionId:", query_execution_id)

while True:
    result = athena.get_query_execution(QueryExecutionId=query_execution_id)
    state = result["QueryExecution"]["Status"]["State"]
    print("State:", state)

    if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
        break

    time.sleep(2)

if state != "SUCCEEDED":
    reason = result["QueryExecution"]["Status"].get("StateChangeReason", "")
    raise RuntimeError(f"Query failed: {reason}")

results = athena.get_query_results(QueryExecutionId=query_execution_id)

import csv

rows = results["ResultSet"]["Rows"]

with open("athena_result.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    for row in rows:
        values = [col.get("VarCharValue", "") for col in row["Data"]]
        writer.writerow(values)

print("结果已保存到 athena_result.csv")