from models.swinunet import build_model as SwinUNet
from models.swinunet_r3 import build_model as SwinUNetR3
from models.unet import build_model as UNet
from models.unet_r3 import build_model as UNetR3
from models.vmunet import build_model as VMUNet
from models.vmunet_r3 import build_model as VMUNetR3

MODELS = {
    "SwinUNet": SwinUNet,
    "SwinUNetR3": SwinUNetR3,
    "VMUNet": VMUNet,
    "VMUNetR3": VMUNetR3,
    "UNet": UNet,
    "UNetR3": UNetR3,
}