# Edge Vision-Language Model (Moondream)

This repository contains the `Moondream` vision-language model, designed to generate captions for images. It utilizes a lightweight, experimental vision encoder and a language model for generating descriptions of input images.

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
```

## Disclaimer

Please note that this model is **small and experimental**. It was created for testing and exploration purposes rather than for achieving state-of-the-art performance. While it can generate image captions, its capabilities are limited compared to larger, more advanced models. We encourage you to explore and experiment, but keep in mind that the results may not match those of high-end, production-ready models.
