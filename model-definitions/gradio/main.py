import random

import gradio
import numpy as np


def greet(img: np.ndarray) -> int:
    return random.randint(0, 9)


if __name__ == "__main__":
    demo = gradio.Interface(
        fn=greet,
        inputs=["image"],
        outputs=["number"],
    )
    demo.launch()  # port=7860
