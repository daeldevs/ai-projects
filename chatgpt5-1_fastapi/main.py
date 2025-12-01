from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openai import OpenAI
from dotenv import load_dotenv
import base64
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/", response_class=HTMLResponse)
def home():
    with open("chat.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/chat")
async def chat(
    message: str = Form(""),
    image: UploadFile | None = File(None)
):

    content = []

    # Add text input
    if message:
        content.append({
            "type": "input_text",
            "text": message
        })

    # Add image input
    if image:
        img_bytes = await image.read()
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        # MUST use input_image + image_url
        content.append({
            "type": "input_image",
            "image_url": f"data:{image.content_type};base64,{b64}"
        })

    # Send to OpenAI Responses API
    response = client.responses.create(
        model="gpt-5.1",
        input=[{
            "role": "user",
            "content": content
        }]
    )

    reply = response.output_text

    return {"reply": reply}
