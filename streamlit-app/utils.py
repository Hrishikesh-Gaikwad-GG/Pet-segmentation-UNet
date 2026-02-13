import torch
import numpy as np
import cv2
from torchvision import transforms

DEVICE = 'cpu'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

def preprocess_image(image):
    return transform(image).unsqueeze(0).to(DEVICE)

def predict_mask(model,image_np, image_tensor, threshold = 0.5):
    with torch.no_grad():
        pred = model(image_tensor)
        pred = torch.sigmoid(pred)
        mask = (pred > threshold).float()

    mask = mask.squeeze().cpu().numpy()
    mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
    return mask_resized

def create_overlay(image_np ,mask_resized):
    overlay = image_np.copy()
    overlay[mask_resized > 0.5] = [255, 0, 0]
    return overlay