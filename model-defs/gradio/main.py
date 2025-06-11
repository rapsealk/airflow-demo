from http import HTTPStatus
from pathlib import Path

import gradio
import numpy as np
import torch
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from torchvision import transforms

from models import Net

model = Net()
model.load_state_dict(torch.load(Path(__file__).parent.parent.parent / "mnist_cnn.ckpt"))  #, map_location=torch.device("cpu")))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Grayscale(),
])

app = FastAPI()

@app.get("/health", status_code=HTTPStatus.OK)
async def health_check() -> Response:
    return JSONResponse({"healthy": True})


def fn(img: np.ndarray) -> int:
    x = transform(img).unsqueeze(0)  # Add batch dimension
    x = model(x)
    x = x.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
    return int(x.item())  # Convert tensor to int and return


gradio_app = gradio.Interface(
    fn=fn,
    inputs=["image"],
    outputs=["number"],
)

app = gradio.mount_gradio_app(app, gradio_app, path="/")
