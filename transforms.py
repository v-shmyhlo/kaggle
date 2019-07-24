import torchvision
from PIL import Image


def transpose(image):
    if not torchvision.transforms.functional._is_pil_image(image):
        raise TypeError('image should be PIL Image. Got {}'.format(type(image)))

    return image.transpose(Image.TRANSPOSE)
