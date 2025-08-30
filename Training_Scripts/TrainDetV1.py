import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B1_Weights
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import json
import time
import torchvision.transforms as T
from dotenv import load_dotenv

load_dotenv()

### HYPERPARAMS
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
WEIGHTS = None
IMG_SIZE = 256
BATCH_SIZE = 32
IMG_DIR = os.getenv("DET_TRAIN_IMAGE_DIR")
LBL_DIR = os.getenv("DET_TRAIN_LABEL_DIR")

### Global
grid_size = 0


### Helper Functions
def compute_batch_iou(preds, targets):
    preds_reshaped = preds.permute(0, 2, 3, 1).reshape(-1, 4)
    targets_reshaped = targets.permute(0, 2, 3, 1).reshape(-1, 4)

    pred_x1 = preds_reshaped[:, 0] - preds_reshaped[:, 2] / 2
    pred_y1 = preds_reshaped[:, 1] - preds_reshaped[:, 3] / 2
    pred_x2 = preds_reshaped[:, 0] + preds_reshaped[:, 2] / 2
    pred_y2 = preds_reshaped[:, 1] + preds_reshaped[:, 3] / 2

    targ_x1 = targets_reshaped[:, 0] - targets_reshaped[:, 2] / 2
    targ_y1 = targets_reshaped[:, 1] - targets_reshaped[:, 3] / 2
    targ_x2 = targets_reshaped[:, 0] + targets_reshaped[:, 2] / 2
    targ_y2 = targets_reshaped[:, 1] + targets_reshaped[:, 3] / 2

    inter_x1 = torch.max(pred_x1, targ_x1)
    inter_y1 = torch.max(pred_y1, targ_y1)
    inter_x2 = torch.min(pred_x2, targ_x2)
    inter_y2 = torch.min(pred_y2, targ_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    pred_area = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
    targ_area = (targ_x2 - targ_x1).clamp(0) * (targ_y2 - targ_y1).clamp(0)
    union_area = pred_area + targ_area - inter_area + 1e-6
    iou = inter_area / union_area
    return iou.mean()

### MODEL DEFINITION
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

class DetectionModel(nn.Module):
    def __init__(self, weights=None, device=None, img_size=256, decode_preds=False):
        super().__init__()
        global grid_size
        grid_size = img_size // 32
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

### Dataset
class DetectionDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size, transform):
        self.img_size = img_size
        global grid_size
        self.grid_size = grid_size
        self.stride = img_size / self.grid_size
        self.img_names = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.lbl_names = sorted([os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir)])
        self.transform = transform
        self.lbls = []
        for idx, pth in enumerate(self.lbl_names):
            labels = []
            with open(pth, 'r') as rf:
                for line in rf:
                    label = list(map(float, line.strip().split(" ")))
                    labels.append(label)
            self.lbls.append(labels)
            print(f"Processing Data... ({idx+1}/{len(self.lbl_names)})", end='\r')
        print("\n")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = Image.open(self.img_names[idx]).convert("RGB")
        labels = self.lbls[idx]
        if self.transform:
            img = self.transform(img)
        bbox_grid = torch.zeros((self.grid_size, self.grid_size, 4), dtype=torch.float32)
        obj_grid = torch.zeros((self.grid_size, self.grid_size, 1), dtype=torch.float32)
        for lbl in labels:
            cx_norm, cy_norm, w_norm, h_norm = lbl
            cx_pixel = cx_norm * self.img_size
            cy_pixel = cy_norm * self.img_size
            w_pixel = w_norm * self.img_size
            h_pixel = h_norm * self.img_size
            grid_x = int(cx_pixel / self.stride)
            grid_y = int(cy_pixel / self.stride)
            if grid_x < self.grid_size and grid_y < self.grid_size:
                offset_x = (cx_pixel % self.stride) / self.stride
                offset_y = (cy_pixel % self.stride) / self.stride
                w_scaled = w_pixel / self.stride
                h_scaled = h_pixel / self.stride
                bbox_grid[grid_y, grid_x] = torch.tensor([offset_x, offset_y, w_scaled, h_scaled])
                obj_grid[grid_y, grid_x] = 1.0
        return img, bbox_grid, obj_grid

### Loss
class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox_loss = nn.SmoothL1Loss(reduction='sum')
        self.obj_loss = nn.BCEWithLogitsLoss(reduction='sum')
    def forward(self, preds, bbox_target, obj_target):
        bbox_pred = preds[:, :4, :, :]
        obj_pred = preds[:, 4:5, :, :]
        bbox_targets = bbox_target.permute(0, 3, 1, 2)
        obj_targets = obj_target.permute(0, 3, 1, 2)
        bbox_loss = self.bbox_loss(bbox_pred * obj_targets, bbox_targets * obj_targets)
        obj_loss = self.obj_loss(obj_pred, obj_targets)
        iou = compute_batch_iou(bbox_pred, bbox_targets)
        valid_cells = obj_targets.sum()
        num_cells = bbox_pred.shape[2] * bbox_pred.shape[3]
        if valid_cells == 0:
            return obj_loss / (bbox_pred.shape[0] * num_cells)
        return (bbox_loss / valid_cells + obj_loss / (bbox_pred.shape[0] * num_cells)) - 0.1 * iou

### Logger
class TrainingLogger():
    def __init__(self, save_dir="Training_Logs", save_file_name="logs.json", clear_save_file_upon_class_instantiation=False):
        self.save_dir = save_dir
        self.save_file_name = save_file_name
        self.save_path = os.path.join(save_dir, save_file_name)
        self.training_data = {}
        os.makedirs(self.save_dir, exist_ok=True)
        if clear_save_file_upon_class_instantiation:
            with open(self.save_path, 'w'): pass
    def log_epoch(self, epoch_number:int, epoch_data:dict):
        data = {f"Epoch: {epoch_number}": epoch_data}
        with open(self.save_path, 'a') as wf:
            json.dump(data, wf, indent=4)
        self.training_data[f"Epoch: {epoch_number}"] = epoch_data
    def save_all_training_data(self, save_path):
        with open(save_path, 'w') as wf:
            json.dump(self.training_data, wf, indent=4)

### Training
def train_model(model, epochs, optimizer, loss_fn, train_dl, device, logger:TrainingLogger):
    model = model.to(device)
    best_loss = float("inf")
    best_loss_epoch = 0
    previous_loss = None
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0
        for i, (img, bbox_grid, obj_grid) in enumerate(train_dl):
            batch_start = time.time()
            img = img.to(device)
            bbox_grid = bbox_grid.to(device)
            obj_grid = obj_grid.to(device)
            pred = model(img)
            loss = loss_fn(pred, bbox_grid, obj_grid)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batches_remaining = len(train_dl) - (i + 1)
            eta = batches_remaining * batch_time
            loss_per_sample = loss.item() / img.size(0)
            print(f"Epoch: {epoch+1} | Batch: {i+1}/{len(train_dl)} | Loss: {loss_per_sample:.4f} | Time Elapsed: {time.time()-start_time:.2f}s | ETA: {eta:.2f}s", end='\r')
        print("\n")
        epoch_avg_loss = epoch_loss / len(train_dl)
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            best_loss_epoch = epoch + 1
        loss_diff = epoch_avg_loss - previous_loss if previous_loss is not None else 0.0
        print(f"Epoch {epoch+1} Summary | Avg Loss: {epoch_avg_loss:.4f} | Prev Loss: {previous_loss} | Loss Dif: {loss_diff:.4f} | Best Epoch: {best_loss_epoch}, Best Loss: {best_loss:.4f}\n")
        previous_loss = epoch_avg_loss
        epoch_data = {
            "Average Loss": epoch_avg_loss,
            "Best Epoch": best_loss_epoch
        }
        logger.log_epoch(epoch+1, epoch_data)

if __name__ == "__main__":
    model = DetectionModel(weights=WEIGHTS, device='cuda', img_size=IMG_SIZE, decode_preds=False)
    loss_fn = DetectionLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomGrayscale(p=0.1),
        T.RandomAutocontrast(p=0.2),
        T.RandomEqualize(p=0.2),
        T.RandomPosterize(bits=4, p=0.05),
        T.RandomSolarize(threshold=128, p=0.05),
        T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
        T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    train_data = DetectionDataset(
        img_dir=IMG_DIR,
        lbl_dir=IMG_DIR,
        img_size=IMG_SIZE,
        transform=train_transform
    )
    train_dl = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    logger = TrainingLogger(save_dir="Detection_Logs", save_file_name="Logs1.json", clear_save_file_upon_class_instantiation=True)
    train_model(model=model, epochs=EPOCHS, optimizer=optimizer, loss_fn=loss_fn, train_dl=train_dl, device='cuda', logger=logger)
