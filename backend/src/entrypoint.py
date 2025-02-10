import base64

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from modules.image_analyser_gemini import process_base64_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/genai-image-analysis")
async def genai_image_analysis(
    image: UploadFile = File(...), productType: str = Form(...), desiredInfo: str = Form(...)
) -> dict[str, str]:
    print(f"Received image file: {image.filename}")
    print(f"Product type: {productType}")
    print(f"Desired info: {desiredInfo}")

    image_data = await image.read()
    input_b64 = base64.b64encode(image_data).decode("utf-8")

    processed_b64 = process_base64_image(input_b64, productType, desiredInfo)

    result_image = f"data:{image.content_type};base64,{processed_b64}"

    return {"resultImage": result_image}
