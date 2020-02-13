from PIL import ImageOps


class Invert(object):
    def __call__(self, input):
        return invert(input)


def invert(input):
    return ImageOps.invert(input)
