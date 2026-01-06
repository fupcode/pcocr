from fastapi import FastAPI
from pcocr.api import ocr


def create_app():
    app = FastAPI(title="OCR Service")

    # 注册子模块
    app.include_router(ocr.router)

    @app.get("/")
    async def root():
        return {"message": "OCR Service is running!"}

    return app
