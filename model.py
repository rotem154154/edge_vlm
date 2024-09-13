
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torchvision.transforms as transforms
import types
import sys
sys.path.insert(0,'ml-mobileclip/')
import mobileclip

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16

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

    def forward(self, images, tokens):
        img_embs = self.vision_encoder(images)
        tok_embs = self.text_model.get_input_embeddings()(tokens)
        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)
        outputs = self.text_model(inputs_embeds=inputs_embeds)
        return outputs
