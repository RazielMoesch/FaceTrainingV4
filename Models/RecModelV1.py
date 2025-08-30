import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from torchvision.models import EfficientNet_B1_Weights




class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

class ArcFace(nn.Module):
    def __init__(self, emb_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weights)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = self.s * (one_hot * target_logits + (1 - one_hot) * cosine)
        return output

class RecognitionModel(nn.Module):
    def __init__(self, emb_dim=1024, num_classes=None, s=30.0, m=0.50, weights="DEFAULT", device="cpu"):
        super().__init__()
        self.device = device
        self.backbone = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1).features.to(self.device)
        self.neck = ConvBNAct(1280, emb_dim, kernel_size=1).to(self.device)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(0.5),
        ).to(self.device)
        self.arcface = ArcFace(emb_dim, num_classes, s=s, m=m).to(self.device) if num_classes is not None else None
        if weights != "DEFAULT" and weights != None:
            self.load_weights_partial(weights)
        else:
            if weights != None:
                self.load_weights_partial("approved_weights/best_recognition.pth")
        self.to(self.device)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = F.normalize(x, dim=1)
        if labels is not None and self.arcface is not None:
            logits = self.arcface(x, labels)
            return logits
        return x
