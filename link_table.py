import os
import boto3
import pandas as pd
from sqlalchemy import create_engine

session = boto3.Session(profile_name="AWSPowerUserAccess-579289528406")
ssm = session.client("ssm")

param_resp =ssm.get_parameter(
    Name="/rds/nmt/db_engine/internal",
    WithDecryption=True
)
conn_url =param_resp["Parameter"]["Value"]
engine =create_engine(conn_url)

print("正在查询数据")
callout_df = pd.read_sql("SELECT * FROM nmt_callout LIMIT 500;",engine)
print(f"nmt_callout 表数据共{len(callout_df)} 行")
print(callout_df.head())
 
# 保存到 CSV 文件
output_dir = r"C:\git"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "nmt_callout.csv")
callout_df.to_csv(output_file, index=False)
print(f"数据已保存到文件: {output_file}")

# 保存为 Excel 文件
output_dir = r"C:\git"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "nmt_callout.xlsx")
 
callout_df.to_excel(output_file, index=False, engine="openpyxl")
print(f"数据已保存到文件: {output_file}")