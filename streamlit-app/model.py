import torch
import segmentation_models_pytorch as smp
from pathlib import Path


DEVICE = 'cpu'
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'

def load_model(model_path = MODEL_DIR / 'best_unet_resnet.pth'):
    model = smp.Unet(
        encoder_name = 'resnet34',
        encoder_weights = None,
        in_channels = 3,
        classes = 1,
        activation = None
    )

    state = torch.load(model_path, map_location = DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model