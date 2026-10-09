import os
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def otsu_threshold(image: np.ndarray) -> Tuple[int, np.ndarray]:
    if image.ndim > 2:
        raise ValueError("输入图像必须是灰度或一维数组")

    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    total_pixels = image.size
    if total_pixels == 0:
        raise ValueError("空数组")

    max_variance = 0.0
    optimal_threshold = 0

    for threshold in range(1, 255):
        w0 = np.sum(hist[:threshold]) / total_pixels
        w1 = np.sum(hist[threshold:]) / total_pixels
        if w0 <= 0 or w1 <= 0:
            continue
        mu0 = np.sum(np.arange(threshold) * hist[:threshold]) / (w0 * total_pixels)
        mu1 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / (w1 * total_pixels)
        variance = w0 * w1 * (mu0 - mu1) ** 2
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = threshold

    if max_variance == 0:
        optimal_threshold = int(np.round(np.mean(image)))

    binary_image = (image > optimal_threshold).astype(np.uint8) * 255
    return optimal_threshold, binary_image


def _otsu_on_values(values: np.ndarray) -> Tuple[float, np.ndarray]:
    if values.size == 0:
        return 0.0, np.zeros(0, dtype=bool)
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmin == vmax:
        return vmin, np.zeros_like(values, dtype=bool)
    scaled = ((values - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
    th_uint8, binary_img = otsu_threshold(scaled)
    threshold_original = vmin + (th_uint8 / 255.0) * (vmax - vmin)
    return float(threshold_original), (binary_img > 0)


def _remove_outliers(values: np.ndarray, method: str = "mad", thresh: float = 3.5) -> np.ndarray:
    if values.size == 0:
        return np.ones_like(values, dtype=bool)
    if method == "mad":
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if mad == 0:
            std = float(np.std(values))
            if std == 0:
                return np.ones_like(values, dtype=bool)
            z = (values - float(np.mean(values))) / std
            return np.abs(z) <= thresh
        modified_z = 0.6745 * (values - med) / mad
        return np.abs(modified_z) <= thresh
    if method == "zscore":
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std == 0:
            return np.ones_like(values, dtype=bool)
        z = (values - mean) / std
        return np.abs(z) <= thresh
    if method == "iqr":
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return np.ones_like(values, dtype=bool)
        lower = q1 - thresh * iqr
        upper = q3 + thresh * iqr
        return (values >= lower) & (values <= upper)
    if method == "percentile":
        p = float(thresh)
        if p <= 0 or p >= 50:
            return np.ones_like(values, dtype=bool)
        lower = np.percentile(values, p)
        upper = np.percentile(values, 100.0 - p)
        return (values >= lower) & (values <= upper)
    return np.ones_like(values, dtype=bool)


def determine_running_by_current(current: np.ndarray,
                                 outlier_method: str = "mad",
                                 outlier_thresh: float = 3.5,
                                 min_frac: float = 0.02,
                                 max_frac: float = 0.98) -> Tuple[float, np.ndarray]:
    """
    用 motorcurrent1avg 判定运行态。先剔除电流异常值，再用 Otsu（或后备分位）得到阈值,
    最终返回阈值和布尔 running_mask（与原数组等长）。
    """
    if current.size == 0:
        raise ValueError("current 为空")
    keep = _remove_outliers(current, method=outlier_method, thresh=outlier_thresh)
    current_clean = current[keep]
    if current_clean.size == 0:
        raise ValueError("motorcurrent1avg 全部被视为异常值")
    th_clean, mask_clean = _otsu_on_values(current_clean)
    high_frac = float(np.mean(mask_clean)) if mask_clean.size > 0 else 0.0
    if high_frac < min_frac or high_frac > max_frac:
        th_clean = float(np.percentile(current_clean, 40))
    running_mask = current >= th_clean
    return th_clean, running_mask


def choose_vibration_threshold(vibration: np.ndarray,
                               running_mask: np.ndarray,
                               min_precision: float = 0.9,
                               min_recall: float = 0.95,
                               safety_abs: float = 0.0,
                               outlier_method: str = "mad",
                               outlier_thresh: float = 3.5) -> Tuple[float, float, float, float]:
    """
    在由电流判定的运行样本上进行振动阈值选择：
      - 在运行样本上剔除异常值
      - 用 Otsu 在运行样本上分簇，取高簇为运行簇（备用：取高50%作为运行簇）
      - 从运行簇左侧（cmin）开始向右测试候选阈值（细分若干步），选择最小同时满足 precision/recall 的阈值
    返回 (chosen_threshold, cmin, p5, otsu_th_on_running_all)
    """
    if vibration.size == 0 or running_mask.size != vibration.size:
        raise ValueError("vibration 和 running_mask 大小不一致或为空")

    if np.sum(running_mask) == 0:
        raise ValueError("没有运行态样本")

    vibration_running = vibration[running_mask]
    keep = _remove_outliers(vibration_running, method=outlier_method, thresh=outlier_thresh)
    if np.any(keep):
        vr_clean = vibration_running[keep]
    else:
        vr_clean = vibration_running.copy()

    otsu_th_run, cluster_mask_run = _otsu_on_values(vr_clean)
    run_cluster = vr_clean[cluster_mask_run]
    if run_cluster.size == 0:
        # 退化：取运行样本中较大的 50% 作为运行簇
        run_cluster = vr_clean[vr_clean >= np.percentile(vr_clean, 50)]

    if run_cluster.size == 0:
        # 最后退化为整个运行样本
        run_cluster = vr_clean

    cmin = float(np.min(run_cluster))
    p5 = float(np.percentile(run_cluster, 5))

    # 生成候选阈值：从 cmin 到 run_cluster 中位数，细分若干步；确保包含 cmin 和 p5
    upper_base = float(np.percentile(run_cluster, 50))
    if upper_base <= cmin:
        upper_base = float(np.max(run_cluster))
    candidates = np.unique(np.concatenate((
        np.array([max(0.0, cmin - safety_abs), cmin, p5]),
        np.linspace(max(0.0, cmin - safety_abs), upper_base, num=200)
    )))
    candidates.sort()

    total_running = float(np.sum(running_mask))
    chosen = None
    for t in candidates:
        pred = vibration >= t
        tp = float(np.sum(pred & running_mask))
        pred_count = float(np.sum(pred))
        recall = tp / total_running if total_running > 0 else 0.0
        precision = tp / pred_count if pred_count > 0 else 0.0
        if (recall >= min_recall) and (precision >= min_precision):
            chosen = float(t)
            break

    if chosen is None:
        # 如果未找到满足条件的阈值，采用保守策略：取运行簇 10% 分位或 cmin（取较大者）
        chosen = float(max(cmin, np.percentile(run_cluster, 10)))

    return chosen, cmin, p5, float(otsu_th_run)


def load_sensor_data(csv_path: str,
                     event_col: str = "event_time_shanghai",
                     vib_col: str = "gbvibforwardrms",
                     current_col: str = "motorcurrent1avg") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {event_col, vib_col, current_col}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 必须包含列: {required}")
    df[event_col] = pd.to_datetime(df[event_col], errors="coerce")
    return df[[event_col, vib_col, current_col]].dropna()


def analyze_thresholds(df: pd.DataFrame,
                       event_col: str = "event_time_shanghai",
                       vib_col: str = "gbvibforwardrms",
                       current_col: str = "motorcurrent1avg",
                       outlier_method: str = "mad",
                       outlier_thresh: float = 3.5,
                       safety_abs: float = 0.0,
                       min_precision: float = 0.9,
                       min_recall: float = 0.95):
    current = df[current_col].to_numpy(dtype=float)
    vibration = df[vib_col].to_numpy(dtype=float)

    current_threshold, running_mask = determine_running_by_current(
        current,
        outlier_method=outlier_method,
        outlier_thresh=outlier_thresh
    )

    vibration_threshold, cmin, lower5, vib_otsu_threshold = choose_vibration_threshold(
        vibration,
        running_mask,
        min_precision=min_precision,
        min_recall=min_recall,
        safety_abs=safety_abs,
        outlier_method=outlier_method,
        outlier_thresh=outlier_thresh
    )

    tp = np.sum((vibration >= vibration_threshold) & running_mask)
    recall = tp / np.sum(running_mask) if np.sum(running_mask) > 0 else 0.0
    pred_count = np.sum(vibration >= vibration_threshold)
    precision = tp / pred_count if pred_count > 0 else 0.0

    return {
        "current_threshold": current_threshold,
        "vibration_threshold": vibration_threshold,
        "vibration_otsu_threshold": vib_otsu_threshold,
        "vibration_run_cluster_min": cmin,
        "vibration_run_cluster_lower5": lower5,
        "running_count": int(np.sum(running_mask)),
        "total_count": int(vibration.size),
        "recall": recall,
        "precision": precision,
        "running_mask": running_mask,
    }


def plot_sensor_data_with_threshold(df: pd.DataFrame,
                                    threshold: float,
                                    event_col: str = "event_time_shanghai",
                                    vib_col: str = "gbvibforwardrms",
                                    current_col: str = "motorcurrent1avg",
                                    running_mask: np.ndarray = None,
                                    html_path: str = "gbvib_threshold_result.html"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[event_col],
        y=df[vib_col],
        mode="lines+markers",
        name="gbvibforwardrms",
        marker=dict(size=6),
        line=dict(width=2),
        hovertemplate="time=%{x}<br>gbvibforwardrms=%{y:.3f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df[event_col],
        y=df[current_col],
        mode="lines+markers",
        name="motorcurrent1avg",
        marker=dict(size=6),
        line=dict(width=2),
        yaxis="y2",
        hovertemplate="time=%{x}<br>motorcurrent1avg=%{y:.3f}<extra></extra>"
    ))
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"gbvibforwardrms 阈值 {threshold:.3f}",
        annotation_position="top left"
    )
    if running_mask is not None and running_mask.size == df.shape[0]:
        fig.add_trace(go.Scatter(
            x=df[event_col][running_mask],
            y=df[vib_col][running_mask],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="当前判定为运行态"
        ))
    fig.update_layout(
        title="gbvibforwardrms 和 motorcurrent1avg 时序图",
        xaxis=dict(title=event_col, tickangle=45),
        yaxis=dict(title="gbvibforwardrms", title_font=dict(color="blue"), tickfont=dict(color="blue")),
        yaxis2=dict(
            title="motorcurrent1avg",
            title_font=dict(color="red"),
            tickfont=dict(color="red"),
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        template="plotly_white"
    )
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"已保存交互式 HTML: {html_path}")
    try:
        os.startfile(html_path)
    except OSError:
        pass
    fig.show()


if __name__ == "__main__":
    csv_path = r"C:\git\1.csv"
    event_col = "event_time_shanghai"
    vib_col = "gbvibforwardrms"
    current_col = "motorcurrent1avg"

    df = load_sensor_data(csv_path, event_col=event_col, vib_col=vib_col, current_col=current_col)

    result = analyze_thresholds(
        df,
        event_col=event_col,
        vib_col=vib_col,
        current_col=current_col,
        outlier_method="mad",
        outlier_thresh=3.5,
        safety_abs=0.0,       # 推荐先用 0.0 或很小值
        min_precision=0.90,   # 调高可减少误报
        min_recall=0.95       # 调低可提高灵敏度
    )

    print("电流判断运行态阈值:", result["current_threshold"])
    print("gbvibforwardrms Otsu 阈值（运行样本上）:", result["vibration_otsu_threshold"])
    print("运行簇最小值 Cmin:", result["vibration_run_cluster_min"])
    print("运行簇 5% 下限:", result["vibration_run_cluster_lower5"])
    print("最终 gbvibforwardrms 阈值:", result["vibration_threshold"])
    print("运行态样本数:", result["running_count"], " / ", result["total_count"])
    print("阈值召回:", round(result["recall"], 4))
    print("阈值精度:", round(result["precision"], 4))

    plot_sensor_data_with_threshold(
        df,
        result["vibration_threshold"],
        event_col=event_col,
        vib_col=vib_col,
        current_col=current_col,
        running_mask=result["running_mask"],
        html_path="gbvib_threshold_result.html"
    )