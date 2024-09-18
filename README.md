---
license: apache-2.0
---

# Edge Vision-Language Model (Moondream)

This repository contains the `Moondream` vision-language model, designed to generate captions for images. It utilizes a lightweight, experimental vision encoder and a language model for generating descriptions of input images.

[![Website](https://img.shields.io/badge/Website-Visit%20Site-blue)](https://rotem154154.github.io/)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-blue)](https://huggingface.co/irotem98/edge_vlm)
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-orange)](https://huggingface.co/spaces/irotem98/edge_vlm)

## Installation

1. Clone the repository:

    ```bash
    git clone https://huggingface.co/irotem98/edge_vlm
    cd edge_vlm
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Here is a simple example to load the model, preprocess an image, and generate a caption:

```python
from model import MoondreamModel
import torch

# Load the model and tokenizer
model = MoondreamModel.load_model()
tokenizer = MoondreamModel.load_tokenizer()

# Load and preprocess an image
image_path = 'img.jpg'  # Replace with your image path
image = MoondreamModel.preprocess_image(image_path)

# Generate the caption
caption = MoondreamModel.generate_caption(model, image, tokenizer)
print('Generated Caption:', caption)
