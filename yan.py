"""
Part 2: 读取数据 → 清洗 → 读取阈值 → 以速度为标准求 Precision/Recall/F1
"""

import json
import os
import warnings

import boto3
import numpy as np
import pandas as pd
import pyathena

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

# ============================================================
# 配置
# ============================================================
EQUIPMENT_NUMBERS = ['30227408']
START_DATE = '2026-06-22'
END_DATE = '2026-06-28'
SPEED_MIN_THRESHOLD = 0    # 速度 > 此值算"运行"，==0 算"停机"
SPEED_VALID_RATIO = 0.01  # 正速度占比低于此值则跳过验证
SPEED_UNIQUE_MIN = 3      # 速度唯一值少于此值则认为数据异常
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
# 数据清洗（振动 + 速度）
# ============================================================
def clean_data(df, vib_col='gbvibforwardrms', spd_col='stepbandspeedleftavg'):
    """清洗：去重、去 NaN、去速度负值"""
    df = df.drop_duplicates(subset=['equipmentnumber', 'event_time_shanghai']).reset_index(drop=True)

    vib = df[vib_col].values.astype(float)
    spd = df[spd_col].values.astype(float)

    vib_valid = np.isfinite(vib)
    spd_valid = np.isfinite(spd) & (spd >= 0)
    mask = vib_valid & spd_valid

    removed = len(vib) - mask.sum()
    print(f"  清洗: 总 {len(vib)} 条，剔除 {removed} 条，保留 {mask.sum()} 条")

    return vib[mask], spd[mask], df[mask]


# ============================================================
# 速度数据质量检查
# ============================================================
def check_speed_quality(speed):
    valid = np.isfinite(speed) & (speed >= 0)
    spd = speed[valid]
    total = len(spd)

    if total == 0:
        return {"is_reliable": False, "reason": "速度数据全部为 NaN/Inf/负值", "stats": {}}

    unique_vals = np.unique(spd)
    n_unique = len(unique_vals)
    n_positive = int(np.sum(spd > SPEED_MIN_THRESHOLD))
    n_zero = int(np.sum((spd >= 0) & (spd <= SPEED_MIN_THRESHOLD)))
    ratio_positive = n_positive / total

    stats = {
        "total_valid": total,
        "n_unique": n_unique,
        "unique_vals_sample": unique_vals[:10].tolist(),
        "n_positive": n_positive,
        "n_zero": n_zero,
        "ratio_positive": ratio_positive,
        "min": float(spd.min()),
        "max": float(spd.max()),
    }

    if n_unique < SPEED_UNIQUE_MIN:
        return {"is_reliable": False,
                "reason": f"速度唯一值只有 {n_unique} 个（< {SPEED_UNIQUE_MIN}），疑似传感器故障",
                "stats": stats}
    if ratio_positive < SPEED_VALID_RATIO:
        return {"is_reliable": False,
                "reason": f"正速度样本仅占 {ratio_positive*100:.2f}%，数据不足以验证",
                "stats": stats}

    return {"is_reliable": True, "reason": "", "stats": stats}


# ============================================================
# 评估函数（已知阈值 + 速度做真值）
# ============================================================
def evaluate_by_speed(vibration, speed, threshold):
    # 速度 > 0 = 运行，速度 == 0 = 停机
    true_running = speed > SPEED_MIN_THRESHOLD
    # 振动 > 阈值 = 预测运行
    pred_running = vibration > threshold

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
        "total_valid": len(vibration),
        "true_running_count": int(np.sum(true_running)),
        "true_stopped_count": int(np.sum(~true_running)),
    }


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    VIB_COL = 'gbvibforwardrms'
    SPD_COL = 'stepbandspeedleftavg'

    print(f"========== Part 2: 读取数据 + 阈值验证 ==========")
    print(f"设备: {EQUIPMENT_NUMBERS}")
    print(f"日期: {START_DATE} ~ {END_DATE}")

    # 1. 读取阈值
    eq_tag = "_".join(EQUIPMENT_NUMBERS)
    threshold_file = os.path.join(OUTPUT_DIR, eq_tag, "threshold.json")

    if not os.path.exists(threshold_file):
        print(f"\n❌ 找不到阈值文件: {threshold_file}")
        print(f"   请先运行 Part 1 (quan_part1_kde.py)")
        exit(1)

    with open(threshold_file, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)
    threshold = threshold_data["threshold"]

    print(f"\n[1] 读取阈值: {threshold}")
    print(f"    来源: {threshold_data.get('train_start_date', '?')} ~ {threshold_data.get('train_end_date', '?')}, "
          f"训练数据 {threshold_data.get('train_data_count', '?')} 条")

    # 2. 读取数据
    print(f"\n[2] 正在从 Athena 查询数据……")
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
    print(f"  查询到 {len(df)} 条数据")

    # 3. 清洗数据
    print(f"\n[3] 清洗数据……")
    vib_clean, spd_clean, df_clean = clean_data(df, VIB_COL, SPD_COL)

    # 4. 速度质量检查
    print(f"\n[4] 速度数据质量检查……")
    speed_quality = check_speed_quality(spd_clean)
    stats = speed_quality["stats"]

    if speed_quality["is_reliable"]:
        print(f"  ✅ 速度数据正常")
        print(f"  有效样本: {stats['total_valid']}")
        print(f"  唯一值数量: {stats['n_unique']}")
        print(f"  正速度(>0): {stats['n_positive']} 条")
        print(f"  零速度(=0): {stats['n_zero']} 条")
        print(f"  速度范围: {stats['min']:.3f} ~ {stats['max']:.3f}")
    else:
        print(f"  ⚠️  速度数据不可靠: {speed_quality['reason']}")
        if stats:
            print(f"  速度值示例: {stats.get('unique_vals_sample', [])}")

    # 5. 验证
    print(f"\n[5] 阈值验证……")
    if speed_quality["is_reliable"]:
        result = evaluate_by_speed(vib_clean, spd_clean, threshold)

        print(f"\n{'='*50}")
        print(f"  阈值: {threshold}")
        print(f"  有效样本: {result['total_valid']} 条")
        print(f"  真实运行(速度>0): {result['true_running_count']} 条")
        print(f"  真实停机(速度=0): {result['true_stopped_count']} 条")
        print(f"  ---")
        print(f"  准确率 (Precision): {result['precision']:.4f}  ({result['precision']*100:.2f}%)")
        print(f"  覆盖率 (Recall):    {result['recall']:.4f}  ({result['recall']*100:.2f}%)")
        print(f"  F1-score:           {result['f1_score']:.4f}  ({result['f1_score']*100:.2f}%)")
        print(f"  ---")
        print(f"  TP(振动>阈值 & 速度>0): {result['tp']}")
        print(f"  FP(振动>阈值 & 速度=0): {result['fp']}")
        print(f"  FN(振动≤阈值 & 速度>0): {result['fn']}")
        print(f"  TN(振动≤阈值 & 速度=0): {result['tn']}")
        print(f"{'='*50}")
    else:
        print(f"  ⚠️  已跳过 —— {speed_quality['reason']}")
        print(f"  建议: 检查速度传感器是否正常接入")
