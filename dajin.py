import numpy as np
from typing import Tuple


def otsu_threshold(image: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    使用Otsu算法（大津算法）计算最佳阈值并进行二值化

    参数:
        image: 输入的灰度图像或一维值数组（numpy数组，期望在0-255）

    返回:
        threshold: 最佳阈值（0-255）
        binary_image: 二值化后的图像（0/255）
    """
    if image.ndim > 2:
        raise ValueError("输入图像必须是灰度或一维数组")

    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    total_pixels = image.size

    max_variance = 0
    optimal_threshold = 0

    for threshold in range(256):
        w0 = np.sum(hist[:threshold]) / total_pixels
        w1 = np.sum(hist[threshold:]) / total_pixels

        if w0 == 0 or w1 == 0:
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
    """
    对一维数值数组应用 Otsu（自动缩放到0-255再反映射）。
    返回: (原始域阈值, 布尔掩码 True=高簇)
    """
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        return float(vmin), np.zeros_like(values, dtype=bool)

    scaled = ((values - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    th_uint8, binary_img = otsu_threshold(scaled)
    threshold_original = vmin + (th_uint8 / 255.0) * (vmax - vmin)
    return float(threshold_original), (binary_img > 0)


# 新增：异常值剔除辅助函数
def _remove_outliers(values: np.ndarray, method: str = "mad", thresh: float = 3.5) -> np.ndarray:
    """
    返回布尔掩码，True 表示保留样本。
    method: "mad"（默认）或 "zscore"
    thresh: 阈值（mad 的 modified z cutoff，通常 3.5；zscore 通常 3.0）
    """
    if values.size == 0:
        return np.ones_like(values, dtype=bool)

    if method == "mad":
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if mad == 0:
            # MAD 为 0 时退化为 z-score 策略
            mean = float(np.mean(values))
            std = float(np.std(values))
            if std == 0:
                return np.ones_like(values, dtype=bool)
            z = (values - mean) / std
            return np.abs(z) <= thresh
        modified_z = 0.6745 * (values - med) / mad
        return np.abs(modified_z) <= thresh
    else:
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std == 0:
            return np.ones_like(values, dtype=bool)
        z = (values - mean) / std
        return np.abs(z) <= thresh


# ...existing code...
def _remove_outliers(values: np.ndarray, method: str = "mad", thresh: float = 3.5) -> np.ndarray:
    """
    返回布尔掩码，True 表示保留样本。
    method: "mad"（默认）或 "zscore" 或 "iqr" 或 "percentile"
    thresh:
      - mad: modified z cutoff（默认 3.5）
      - zscore: z cutoff（通常 3.0）
      - iqr: IQR multiplier（例如 1.5）
      - percentile: percent to trim at each tail（例如 1.0 表示去掉上下各1%）
    """
    if values.size == 0:
        return np.ones_like(values, dtype=bool)

    if method == "mad":
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if mad == 0:
            # MAD 为 0 时退化为 z-score 策略
            mean = float(np.mean(values))
            std = float(np.std(values))
            if std == 0:
                return np.ones_like(values, dtype=bool)
            z = (values - mean) / std
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
            # 非法参数时不剔除
            return np.ones_like(values, dtype=bool)
        lower = np.percentile(values, p)
        upper = np.percentile(values, 100.0 - p)
        return (values >= lower) & (values <= upper)

    # 未知方法，返回全保留
    return np.ones_like(values, dtype=bool)


def analyze_two_clusters(csv_path: str, value_col: int = 0, has_header: bool = True,
                         k_sigma: float = 3.0, remove_outliers: bool = True,
                         outlier_method: str = "mad", outlier_thresh: float = 3.5,
                         max_removal_frac: float = 0.1):
    """
    增加参数 max_removal_frac：若剔除数量超过该比例（相对于原始样本数），则放弃剔除以避免误删大量正常簇。
    """
    # 1) 读取数据（只读一次）
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1 if has_header else 0)
    if data.ndim == 1:
        values = data
    else:
        values = data[:, value_col]
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("未能从 CSV 中读取到有效数值")

    # 保留原始拷贝以便在剔除过多时回退
    original_values = values.copy()
    total_before = values.size

    # 1.5) 可选：剔除异常值
    outlier_removed = False
    if remove_outliers:
        keep_mask = _remove_outliers(values, method=outlier_method, thresh=outlier_thresh)
        removed_count = int(np.sum(~keep_mask))
        # 若剔除比例过大，则放弃剔除（防止把一个簇当作异常全部剔除）
        if removed_count > max_removal_frac * total_before:
            # 放弃剔除，保留原始数据
            removed_count = 0
            values = original_values
            outlier_removed = False
        else:
            values = values[keep_mask]
            outlier_removed = (removed_count > 0)
    else:
        removed_count = 0

    if values.size == 0:
        raise ValueError("剔除异常值后没有剩余样本，请调整剔除参数")

    # 2) 用 Otsu 找两簇之间的分界阈值（复用已读数据，不重复读 CSV）
    divide_threshold, high_mask = _otsu_on_values(values)
    low_mask = ~high_mask

    low_values = values[low_mask]
    high_values = values[high_mask]

    # 3) 对每个簇单独计算阈值边界（均值 ± k*标准差）
    def cluster_stats(v):
        if v.size == 0:
            return {"mean": float("nan"), "std": 0.0,
                    "lower": float("nan"), "upper": float("nan"), "count": 0}
        mean, std = float(np.mean(v)), float(np.std(v))
        return {
            "mean": mean,
            "std": std,
            "lower": mean - k_sigma * std,
            "upper": mean + k_sigma * std,
            "count": int(v.size),
        }

    # 判断是否为单簇（Otsu 划分后一侧为空）
    if low_values.size == 0 or high_values.size == 0:
        # 单簇情况
        single_stats = cluster_stats(values)
        n_clusters = 1
        clusters = [single_stats]
        # 兼容旧字段：把 low_cluster 设为单簇，high_cluster 置空
        low_cluster = single_stats
        high_cluster = cluster_stats(np.array([], dtype=values.dtype))
        # masks: 全部归为低簇（兼容）
        low_mask = np.ones_like(values, dtype=bool)
        high_mask = np.zeros_like(values, dtype=bool)
        divide_threshold = float("nan")
    else:
        n_clusters = 2
        low_cluster = cluster_stats(low_values)
        high_cluster = cluster_stats(high_values)
        clusters = [low_cluster, high_cluster]

    return {
        "n_clusters": n_clusters,
        "clusters": clusters,
        "divide_threshold": divide_threshold,
        "low_cluster": low_cluster,
        "high_cluster": high_cluster,
        "values": values,
        "clean_values": values,
        "low_mask": low_mask,
        "high_mask": high_mask,
        "removed_count": removed_count,
        "outlier_removed": outlier_removed,
    }
# ...existing code...


def _print_cluster_summary(result):
    if result["n_clusters"] == 1:
        c = result["clusters"][0]
        print("=" * 50)
        print("检测到单簇 (n=1)")
        print("样本数 :", c["count"])
        print("均值   :", round(c["mean"], 4))
        print("标准差 :", round(c["std"], 4))
        print("阈值范围:", "[", round(c["lower"], 4), ",", round(c["upper"], 4), "]")
        print("剔除的异常样本数 :", result["removed_count"])
        print("=" * 50)
    else:
        print("=" * 50)
        print("检测到两簇 (n=2)")
        print("剔除的异常样本数 :", result["removed_count"])
        print("两簇分界阈值（Otsu）:", round(result["divide_threshold"], 4))
        print("-" * 50)
        low = result["low_cluster"]
        print(f"【低簇 ~{round(low['mean'], 2)}】")
        print("  样本数 :", low["count"])
        print("  均值   :", round(low["mean"], 4))
        print("  标准差 :", round(low["std"], 4))
        print("  阈值范围: [", round(low["lower"], 4), ",", round(low["upper"], 4), "]")
        print("-" * 50)
        high = result["high_cluster"]
        print(f"【高簇 ~{round(high['mean'], 2)}】")
        print("  样本数 :", high["count"])
        print("  均值   :", round(high["mean"], 4))
        print("  标准差 :", round(high["std"], 4))
        print("  阈值范围: [", round(high["lower"], 4), ",", round(high["upper"], 4), "]")
        print("=" * 50)


if __name__ == "__main__":
    # ===== 修改这里：换成你自己的 CSV 路径和列索引 =====
    csv_path = r"C:\git\16-19.csv"
    value_col = 1
    has_header = True
    # ==================================================

    # 可通过参数关闭异常值剔除或改用 zscore
    result = analyze_two_clusters(csv_path, value_col=value_col,
                                  has_header=has_header, k_sigma=3.0,
                                  remove_outliers=True, outlier_method="mad", outlier_thresh=3.5)

    _print_cluster_summary(result)
