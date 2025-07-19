from models.vmunet.vmunet import VMUNet
import os

model_config = {
        'num_classes': 9, 
        'input_channels': 3, 
        'depths': [2,2,2,2],
        'depths_decoder': [2,2,2,1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': os.path.join(os.path.dirname(__file__), 'pretrain/vmamba_small_e238_ema.pth'),
    }

def build_model(**kwargs):
    model_config["num_classes"] = kwargs.get("num_classes", 9)
    model_config["input_channels"] = kwargs.get("in_channels", 3)
    return VMUNet(**model_config)


if __name__ == "__main__":
    print(build_model())