import pyathena
import boto3
import pandas as pd
import numpy as np
import warnings

from kdex import (
    clean_vibration,
    kde_threshold,
    analyze_margin,
    find_current_threshold_kde,
    evaluate,
)

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy"
)


class Athena_Connector:
    def __init__(self):
        self.conn = pyathena.connect(
            s3_staging_dir="s3://athena-query-prod-zhen/",
            session=boto3.Session(
                profile_name="AWSPowerUserAccess-579289528406"
            )
        )

    def query(self, sql):
        return pd.read_sql(sql, self.conn)

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    conn = Athena_Connector()

    sql = """
        SELECT equipmentnumber,
               gbvibforwardrms,
               motorcurrent1avg,
               motorcurrent2avg,
               motorcurrent3avg,
               modeset,
               operationstatus,
               timestamp,
               (
                   FROM_ISO8601_TIMESTAMP(timestamp)
                   AT TIME ZONE 'Asia/Shanghai'
               ) AS event_time_shanghai
        FROM data_cleansed."anyescalator"
        WHERE equipmentnumber IN ('45896520')
          AND eventdate BETWEEN '2026-07-10' AND '2026-07-13'
        ORDER BY timestamp
    """

    try:
        # 直接读取到内存，不保存 CSV
        df = conn.query(sql)
        print(f"读取数据: {len(df)} 行")

        vibration = pd.to_numeric(
            df["gbvibforwardrms"],
            errors="coerce"
        ).to_numpy(dtype=float)

        # 振动数据清洗
        cleaned_vibration, is_valid = clean_vibration(vibration)

        if len(cleaned_vibration) < 10:
            raise ValueError("有效振动数据少于 10 条")

        # 计算振动阈值
        vibration_threshold = kde_threshold(cleaned_vibration)
        margin = analyze_margin(
            cleaned_vibration,
            vibration_threshold
        )

        print("\n振动阈值分析")
        print("-" * 50)
        print(f"KDE 振动阈值: {vibration_threshold:.4f}")
        print(f"有效样本数: {len(cleaned_vibration)}")
        print(f"清洗移除数: {len(vibration) - is_valid.sum()}")

        if "error" not in margin:
            print(f"停机 P99: {margin['stop_p99']:.4f}")
            print(f"运行 P1: {margin['run_p1']:.4f}")
            print(f"最小裕度: {margin['min_margin']:.4f}")

        # 电流验证
        current_cols = [
            "motorcurrent1avg",
            "motorcurrent2avg",
            "motorcurrent3avg",
        ]

        if all(col in df.columns for col in current_cols):
            currents = df[current_cols].apply(
                pd.to_numeric,
                errors="coerce"
            )

            current_mean = currents.mean(axis=1).to_numpy(dtype=float)

            valid_current = np.isfinite(current_mean)
            if valid_current.sum() >= 10:
                equipment_ids = (
                    df["equipmentnumber"].to_numpy()
                    if "equipmentnumber" in df.columns
                    else None
                )

                current_threshold = find_current_threshold_kde(
                    current_mean,
                    equipment_ids=equipment_ids
                )

                # 同时保证振动和电流均有效
                eval_mask = is_valid & valid_current

                result = evaluate(
                    vibration[eval_mask],
                    current_mean[eval_mask],
                    current_threshold,
                    vibration_threshold
                )

                print("\n电流验证")
                print("-" * 50)
                print(f"电流阈值: {current_threshold:.4f}")
                print(f"Precision: {result['precision']:.4f}")
                print(f"Recall:    {result['recall']:.4f}")
                print(f"F1-score:  {result['f1_score']:.4f}")
                print(
                    f"TP={result['tp']} "
                    f"FP={result['fp']} "
                    f"FN={result['fn']} "
                    f"TN={result['tn']}"
                )
            else:
                print("\n有效电流数据不足，跳过电流验证")
        else:
            print("\n缺少电流列，跳过电流验证")

    finally:
        conn.close()