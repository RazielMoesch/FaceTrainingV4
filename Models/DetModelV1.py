import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B1_Weights


# NOTES
# Using B1 for larger Backbone

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

class DetectionModel(nn.Module):
    def __init__(self, weights=None, device=None, img_size=240, decode_preds=True):
        super().__init__()
        self.img_size = img_size
        self.decode_preds = decode_preds
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1).features.to(self.device)
        self.neck = nn.Sequential(
            ConvBNAct(1280, 1000, 3, padding=1),
            ConvBNAct(1000, 800, 3, padding=1),
            ConvBNAct(800, 400, 3, padding=1)
        ).to(self.device)
        self.bbox_head = nn.Sequential(
            nn.Conv2d(400, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 4, 1)
        ).to(self.device)
        self.obj_head = nn.Sequential(
            nn.Conv2d(400, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 1)
        ).to(self.device)
        if weights:
            self.load_state_dict(torch.load(weights, map_location=self.device))
        self.to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        bbox = self.bbox_head(x)
        obj = self.obj_head(x)
        x = torch.cat([bbox, obj], dim=1)
        if self.decode_preds:
            x = self.decode_predictions(x, x.shape[2], x.shape[3])
        return x

    def decode_predictions(self, preds, H, W, conf_thresh=0.01):
        B, _, _, _ = preds.shape
        preds = preds.permute(0, 2, 3, 1).contiguous()
        stride_x = self.img_size / W
        stride_y = self.img_size / H
        all_boxes = []
        for b in range(B):
            pred = preds[b]
            conf = torch.sigmoid(pred[..., 4])
            conf_mask = conf > conf_thresh
            boxes = []
            ys, xs = conf_mask.nonzero(as_tuple=True)
            for y, x in zip(ys, xs):
                px, py, pw, ph = pred[y, x, :4]
                pconf = conf[y, x].item()
                cx = (x + px.item()) * stride_x
                cy = (y + py.item()) * stride_y
                w = pw.item() * stride_x
                h = ph.item() * stride_y
                boxes.append([cx, cy, w, h, pconf])
            all_boxes.append(boxes)
        return all_boxes