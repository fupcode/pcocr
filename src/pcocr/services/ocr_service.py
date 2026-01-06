from pcocr.utils.img import filter_white_text, postprocess_ocr_result
from cnocr import CnOcr
from PIL import ImageFile
import numpy as np


ocr_model = CnOcr(rec_model_name="scene-densenet_lite_136-gru", det_model_name="ch_PP-OCRv3_det")


async def ocr_image(image: ImageFile.ImageFile, color: str):
    """
    使用 cnocr 识别图像中的文字
    :param color:
    :param image: PIL Image
    :return: 识别结果列表
    """

    # 预处理图像，过滤白色文字
    img_arr_original = np.array(image.convert("RGB"), dtype=np.uint8)
    if color == "white":
        img_arr = filter_white_text(img_arr_original)
    else:
        img_arr = image

    # 使用 cnocr 进行 OCR 识别
    ocr_result = ocr_model.ocr(img_arr, resized_shape=(512, 768))

    # 后处理 OCR 结果
    result = postprocess_ocr_result(ocr_result, image.size)

    return result