import numpy as np


def filter_white_text(img_arr: np.ndarray, lower_v=50, upper_s=50) -> np.ndarray:
    """
    仅保留白色文字，但改为“灰度保留”，不二值化。

    返回:
        numpy.ndarray, dtype=np.uint8, shape=(H, W)
        - 白字部分: 灰度值 (保留亮度信息)
        - 非白字部分: 0 (黑色背景)
    """
    if not (isinstance(img_arr, np.ndarray) and img_arr.ndim == 3 and img_arr.shape[2] == 3):
        raise ValueError("输入图像必须是形状为 (H, W, 3) 的 numpy 数组")

    # 计算 HSV 中的亮度 V 和饱和度 S
    r = img_arr[:, :, 0] / 255.0
    g = img_arr[:, :, 1] / 255.0
    b = img_arr[:, :, 2] / 255.0

    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])

    v = maxc * 100  # 亮度
    s = ((maxc - minc) / (maxc + 1e-8)) * 100  # 饱和度

    # 判断哪些区域属于白色文字
    mask = (v >= lower_v) & (s <= upper_s)

    # 生成灰度图（单通道）
    # 常用公式: 0.299R + 0.587G + 0.114B
    gray = (0.299 * img_arr[:, :, 0] +
            0.587 * img_arr[:, :, 1] +
            0.114 * img_arr[:, :, 2]).astype(np.uint8)

    # 其他区域全部变黑
    gray[~mask] = 0

    return gray


def postprocess_ocr_result(ocr_results, image_size, score_threshold=0.0):
    """
    OCR 结果后处理：
    - text -> subtext
    - 去掉低分结果
    - position -> 相对坐标四点形式，字段改名为 box
    - 去掉 score 和 box_type

    :param ocr_results: 原始 OCR 输出列表
    :param image_size: (width, height)
    :param score_threshold: 分数阈值
    :return: 后处理后的结果列表
    """
    width, height = image_size
    processed = []

    for item in ocr_results:
        score = item.get("score", 1.0)
        if score < score_threshold:
            continue

        subtext = item.get("text", "")
        position = item.get("position", None)

        if position is not None:
            # 将四点坐标转成相对坐标
            box = (position / np.array([width, height], dtype=np.float32)).tolist()
        else:
            box = [[0, 0], [0, 0], [0, 0], [0, 0]]

        processed.append({
            "subtext": subtext,
            "box": box
        })

    return processed
