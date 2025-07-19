from models.swinunet.vision_transformer import SwinUnet
from models.swinunet.config import get_config


def build_model(**kwargs):
    config = get_config(kwargs)
    img_size = kwargs.get("img_size", 224)
    num_classes = kwargs.get("num_classes", 9)
    return SwinUnet(config, img_size=img_size, num_classes=num_classes)


if __name__ == "__main__":
    print(build_model())