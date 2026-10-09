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
EQUIPMENT_NUMBERS = ['30227408']
START_DATE = '2026-06-20'
END_DATE = '2026-06-23'
KNOWN_THRESHOLD = 3.02       # ← 你已知的振动阈值，改成你的值
SPEED_MIN_THRESHOLD = 0    # 速度 > 此值才算"运行"（0 = 严格大于0才算运行）
SPEED_VALID_RATIO = 0.01    # 有效速度占比低于此值则跳过验证
SPEED_UNIQUE_MIN = 3        # 速度唯一值少于此值则认为数据异常
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
# 数据清洗
# ============================================================
def clean_vibration(vibration: np.ndarray) -> tuple:
    is_valid = np.isfinite(vibration)
    return vibration[is_valid], is_valid


# ============================================================
# 速度数据质量检查
# ============================================================
def check_speed_quality(speed: np.ndarray) -> dict:
    valid = np.isfinite(speed) & (speed >= 0)
    spd = speed[valid]
    total = len(spd)

    if total == 0:
        return {"is_reliable": False, "reason": "速度数据全部为 NaN/Inf/负值", "stats": {}}

    unique_vals = np.unique(spd)
    n_unique = len(unique_vals)
    n_positive = int(np.sum(spd > SPEED_MIN_THRESHOLD))
    n_negative = int(np.sum(spd < 0))
    n_zero = int(np.sum((spd >= 0) & (spd <= SPEED_MIN_THRESHOLD)))
    ratio_positive = n_positive / total

    stats = {
        "total_valid": total,
        "n_unique": n_unique,
        "unique_vals_sample": unique_vals[:10].tolist(),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero_near": n_zero,
        "ratio_positive": ratio_positive,
        "min": float(spd.min()),
        "max": float(spd.max()),
        "mean": float(spd.mean()),
    }

    if n_unique < SPEED_UNIQUE_MIN:
        return {
            "is_reliable": False,
            "reason": f"速度唯一值只有 {n_unique} 个（< {SPEED_UNIQUE_MIN}），疑似传感器未接入或数据通道故障",
            "stats": stats,
        }
    if n_negative == total:
        return {
            "is_reliable": False,
            "reason": "速度全部为负值，数据无效",
            "stats": stats,
        }
    if ratio_positive < SPEED_VALID_RATIO:
        return {
            "is_reliable": False,
            "reason": f"正速度样本仅占 {ratio_positive*100:.2f}%（< {SPEED_VALID_RATIO*100}%），数据不足以验证",
            "stats": stats,
        }

    return {"is_reliable": True, "reason": "", "stats": stats}


# ============================================================
# 评估函数（已知阈值 + 速度做真值）
# ============================================================
def evaluate_by_speed(vibration, speed, threshold):
    """
    用速度作为真值，评估已知振动阈值的表现。
    三态逻辑：
      speed < 0      → 剔除（不参与计算）
      speed == 0     → 停机
      speed > 0      → 运行
    振动判断：
      vib > threshold → 预测为运行
      vib <= threshold → 预测为停机
    """
    valid = np.isfinite(speed) & np.isfinite(vibration) & (speed >= 0)
    vib_v, spd_v = vibration[valid], speed[valid]

    true_running = spd_v > SPEED_MIN_THRESHOLD
    pred_running = vib_v > threshold

    tp = int(np.sum(pred_running & true_running))
    fp = int(np.sum(pred_running & ~true_running))
    fn = int(np.sum(~pred_running & true_running))
    tn = int(np.sum(~pred_running & ~true_running))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_valid": len(vib_v),
        "true_running_count": int(np.sum(true_running)),
        "true_stopped_count": int(np.sum(~true_running)),
    }


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    VIBRATION_COL = 'gbvibforwardrms'
    SPEED_COL = 'stepbandspeedleftavg'

    print(f"设备: {EQUIPMENT_NUMBERS}")
    print(f"日期: {START_DATE} ~ {END_DATE}")

    # 1. 查询数据
    print("\n正在从 Athena 查询数据……")
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
        combined = conn.query(sql)
    finally:
        conn.close()

    # 去重
    dedup_cols = ['equipmentnumber', 'event_time_shanghai']
    combined = combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

     # 2. 准备数据
    vibration_all = combined[VIBRATION_COL].values.astype(float)
    speed_all = combined[SPEED_COL].values.astype(float)

    # 剔除无效值
    eval_mask = np.isfinite(vibration_all) & np.isfinite(speed_all) & (speed_all >= 0)
    vib_clean = vibration_all[eval_mask]
    spd_clean = speed_all[eval_mask]
    removed = len(vibration_all) - len(vib_clean)

    # 3. 速度数据质量检查
    speed_quality = check_speed_quality(spd_clean)

    # 4. 输出
    #print(f"\n{'='*60}")
    #print(f"设备号: {EQUIPMENT_NUMBERS}")
    #print(f"  数据量: {len(combined)} 条")
    #print(f"  剔除无效(振动NaN/速度负值): {removed} 条")
    #print(f"  有效数据: {len(vib_clean)} 条")
    #print(f"{'='*60}")

    # 速度数据质量报告
    print(f"\n速度数据质量检查:")
    stats = speed_quality["stats"]
    if speed_quality["is_reliable"]:
        print(f"  ✅ 速度数据正常")      
    else:
        print(f"  ⚠️  速度数据不可靠: {speed_quality['reason']}")

    # 5. 已知阈值验证
    if speed_quality["is_reliable"]:
        result = evaluate_by_speed(vib_clean, spd_clean, KNOWN_THRESHOLD)

        print(f"  振动阈值: {KNOWN_THRESHOLD}")
        print(f"  有效样本: {result['total_valid']} 条")
        print(f"  真实运行(速度>0): {result['true_running_count']} 条")
        print(f"  真实停机(速度=0): {result['true_stopped_count']} 条")
        print(f"  准确率 (Precision): {result['precision']:.4f} ")
        print(f"  覆盖率 (Recall):    {result['recall']:.4f} ")
        print(f"  F1-score:           {result['f1_score']:.4f} ")
    else:
        print(f"\n验证结果:")
        print(f"  ⚠️  已跳过 —— {speed_quality['reason']}")
        print(f"  建议: 检查速度传感器是否正常接入，或换用其他字段验证")