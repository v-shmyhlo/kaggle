import torch

COLORS = torch.tensor([
    [19, 0, 249],
    [42, 255, 31],
    [255, 0, 25],
    [45, 255, 252],
    [250, 0, 253],
    [254, 255, 40],
], dtype=torch.float) / 255

RANGES = torch.tensor([
    [0, 51],
    [0, 107],
    [0, 64],
    [0, 191],
    [0, 89],
    [0, 191],
], dtype=torch.float) / 255


# def images_to_rgb(input, clamp=False):
#     colors, ranges = COLORS.to(input.device), RANGES.to(input.device)
#
#     min, max = ranges[:, 0], ranges[:, 1]
#     min, max = min.view(1, 6, 1, 1), max.view(1, 6, 1, 1)
#     input = input / (max - min) + min
#     if clamp:
#         input = input.clamp(0., 1.)
#
#     colors = colors.reshape((1, 6, 3, 1, 1))
#     input = input.unsqueeze(2)
#     input = input * colors
#     input = input.mean(1)
#
#     return input

def images_to_rgb(input):
    colors, ranges = COLORS.to(input.device), RANGES.to(input.device)

    min, max = input.min(), input.max()
    input = (input - min) / (max - min)

    colors = colors.reshape((1, 6, 3, 1, 1))
    input = input.unsqueeze(2)
    input = input * colors
    input = input.mean(1)

    return input
