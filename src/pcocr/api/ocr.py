from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pcocr.services import ocr_service
from PIL import Image
import logging
import io

router = APIRouter(prefix="/ocr", tags=["ocr"])
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@router.get("")
@router.get("/")
def get_ocr():
    return {"message": "Send a POST request to this URL to recognize the image."}


@router.post("")
@router.post("/")
async def post_ocr(file: UploadFile = File(...), type: str = Form(...)):
    # 读取文件内容
    contents = await file.read()

    # 转为PIL Image对象
    image = Image.open(io.BytesIO(contents))

    # 使用OCR进行图片识别
    try:
        anchors = await ocr_service.ocr_image(image, type)
    except Exception as e:
        logger.error("OCR processing error: %s", str(e))
        raise HTTPException(status_code=500, detail="OCR processing failed.")

    return {"anchors": anchors}
