from models.unet.unet_model import UNet



def build_model(**kwargs):
    img_size = kwargs.get("img_size", 224)
    num_classes = kwargs.get("num_classes", 9)
    return UNet(n_channels=3, n_classes=num_classes)


if __name__ == "__main__":
    model = build_model()
    model.flops()