from pathlib import Path

import gradio
import numpy as np
import torch
from torchvision import transforms

# from ...dags.mnist.v1.models import Net  # noqa: E402
from models import Net

model = Net()
# torch.load()
# model.load_state_dict(torch.load("model-defs/gradio/mnist_model.pth"))  #, map_location=torch.device("cpu")))
model.load_state_dict(torch.load(Path(__file__).parent.parent.parent / "mnist_cnn.ckpt"))  #, map_location=torch.device("cpu")))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Grayscale(),
])


def greet(img: np.ndarray) -> int:
    x = transform(img).unsqueeze(0)  # Add batch dimension
    x = model(x)
    x = x.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
    return int(x.item())  # Convert tensor to int and return


if __name__ == "__main__":
    demo = gradio.Interface(
        fn=greet,
        inputs=["image"],
        outputs=["number"],
    )
    demo.launch()  # server_port=7860
