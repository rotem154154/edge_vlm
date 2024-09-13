import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torchvision.transforms as transforms
import types
import os
import sys
sys.path.insert(0, 'ml-mobileclip/')
import mobileclip

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16

# Download mobileclip_s2.pt if it doesn't exist locally
if not os.path.exists('mobileclip_s2.pt'):
    model_url = 'https://huggingface.co/irotem98/edge_vlm/resolve/main/mobileclip_s2.pt'
    os.system(f"wget {model_url}")

def split_chessboard(x, num_split):
    B, C, H, W = x.shape
    h, w = H // num_split, W // num_split
    x_split = torch.cat([x[:, :, i*h:(i+1)*h, j*w:(j+1)*w] for i in range(num_split) for j in range(num_split)], dim=0)
    return x_split

def merge_chessboard(x, num_split):
    B, C, H, W = x.shape
    b = B // (num_split**2)
    x_merge = torch.cat([torch.cat([x[(i*num_split + j)*b:(i*num_split + j + 1)*b] for j in range(num_split)], dim=-1)
                         for i in range(num_split)], dim=-2)
    return x_merge

class FeatureIRLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.GELU(approximate='tanh')
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MobileVision(nn.Module):
    def __init__(self):
        super(MobileVision, self).__init__()
        self.vision, _, _ = mobileclip.create_model_and_transforms('mobileclip_s2', pretrained='mobileclip_s2.pt')
        self.vision = self.vision.image_encoder.model.eval().to(DEVICE).half()

        def new_forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.forward_embeddings(x)
            x = self.forward_tokens(x)
            return self.conv_exp(x)
        self.vision.forward = types.MethodType(new_forward, self.vision)

        self.projection = FeatureIRLayer(1280*2, 4096, 1536).to(DEVICE).half()
        self.projection2 = nn.Linear(4096, 1536).to(DEVICE).half()

    def forward(self, x):
        with torch.no_grad():
            resized_img = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
            out1 = self.vision(resized_img)
            x = split_chessboard(x, 2)
            x = self.vision(x)
            x = merge_chessboard(x, 2)
            x = F.interpolate(x, size=(8, 8), mode='area')
            x = torch.cat([out1, x], dim=1)

            x = x.reshape(x.size(0), x.size(1), -1)
            x = x.permute(0, 2, 1)

        x = self.projection(x)
        x = self.projection2(x)
        return x

class MoondreamModel(nn.Module):
    def __init__(self):
        super(MoondreamModel, self).__init__()
        self.vision_encoder = MobileVision()
        self.text_model = AutoModelForCausalLM.from_pretrained(
            "h2oai/h2o-danube3-500m-chat",
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map={"": DEVICE}
        )
        self.load_state_dict(torch.load('moondream_model_state_dict.pt'))

    def forward(self, images, tokens):
        img_embs = self.vision_encoder(images)
        tok_embs = self.text_model.get_input_embeddings()(tokens)
        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)
        outputs = self.text_model(inputs_embeds=inputs_embeds)
        return outputs

    @staticmethod
    def load_model():
        model = MoondreamModel().to(DEVICE).half()
        return model

    @staticmethod
    def load_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained("h2oai/h2o-danube3-500m-chat", trust_remote_code=True)
        return tokenizer

    @staticmethod
    def preprocess_image(image_path, img_size=512):
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(torch.float16)),
        ])
        image = Image.open(image_path).convert('RGB')
        image = transform(image).to(DEVICE)
        return image

    @staticmethod
    def generate_caption(model, image, tokenizer, max_length=128):
        model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(DEVICE)
            img_embs = model.vision_encoder(image)

            generated = [tokenizer.bos_token_id]
            descriptive_prompt = tokenizer(
                f"\n\nDescriptions of the image:",
                add_special_tokens=False
            ).input_ids
            generated.extend(descriptive_prompt)

            for _ in range(max_length):
                input_ids = torch.tensor(generated, dtype=torch.long, device=DEVICE).unsqueeze(0)
                tok_embs = model.text_model.get_input_embeddings()(input_ids)
                inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)
                outputs = model.text_model(inputs_embeds=inputs_embeds)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()

                if next_token == tokenizer.sep_token_id:
                    break

                generated.append(next_token)

            return tokenizer.decode(generated, skip_special_tokens=True)
