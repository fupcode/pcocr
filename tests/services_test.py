from pcocr.services.ocr_service import ocr_image
from PIL import Image, ImageDraw, ImageFile
import asyncio
import numpy as np

# 打开图片
img_path = "images/3.png"
image = ImageFile.Image.open(img_path).convert("RGB")
width, height = image.size

# 异步调用 OCR
result = asyncio.run(ocr_image(image, "white"))

# 创建可绘制对象
draw = ImageDraw.Draw(image)

# 遍历结果画框
for item in result:
    box = item["box"]  # 四点相对坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    subtext = item["subtext"]

    # 转回像素坐标
    pixel_box = [(x*width, y*height) for x, y in box]

    # PIL 的 polygon 画多边形
    draw.polygon(pixel_box, outline="red", width=2)

    # 在左上角标注文字
    draw.text(pixel_box[0], subtext, fill="blue")

# 保存或展示
image.show()  # 弹出窗口显示
image.save("images/3_ocr_debug.png")  # 保存标记后的图片
