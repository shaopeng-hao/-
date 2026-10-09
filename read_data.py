import pyathena
import boto3
import pandas as pd
from datetime import datetime
import os
import warnings

# 屏蔽 pyathena 直连触发的 SQLAlchemy 警告（无害，不影响查询）
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

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


if __name__ == "__main__":
    conn = Athena_Connector()
    sql = """
        SELECT equipmentnumber, gbvibforwardrms, motorcurrent1avg, stepbandspeedleftavg,
               modeset, operationstatus,
               (FROM_ISO8601_TIMESTAMP(timestamp) AT TIME ZONE 'Asia/Shanghai') AS event_time_shanghai
        FROM data_cleansed."anyescalator"
        WHERE equipmentnumber IN ('30536165')
          AND eventdate BETWEEN '2026-07-20' AND '2026-07-26'
        ORDER BY timestamp
    """
    df = conn.query(sql)
    print(df)

    output_dir = r"C:\git"
    os.makedirs(output_dir, exist_ok=True)

    # 保存 CSV
    df.to_csv(os.path.join(output_dir, "test1.csv"), index=False, encoding='utf-8-sig')
    print("已保存: test1.csv")

    conn.close()
