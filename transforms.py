import torchvision
from PIL import Image


def transpose(image):
    if not torchvision.transforms.functional._is_pil_image(image):
        raise TypeError('image should be PIL Image. Got {}'.format(type(image)))

    return image.transpose(Image.TRANSPOSE)


class ApplyTo(object):
    def __init__(self, tos, transform):
        self.tos = tos
        self.transform = transform

    def __call__(self, input):
        input = {
            **input,
            **{to: self.transform(input[to]) for to in self.tos},
        }

        return input
