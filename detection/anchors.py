import math

import torch


def build_anchors_maps(image_size, anchor_levels, p2, p7):
    h, w = image_size
    includes = [p2, True, True, True, True, p7]
    assert len(anchor_levels) == len(includes)

    for _ in range(2):
        h, w = math.ceil(h / 2), math.ceil(w / 2)

    anchor_maps = []
    for anchors, include in zip(anchor_levels, includes):
        if include:
            for anchor in anchors:
                anchor_map = build_anchor_map(image_size, (h, w), anchor)
                anchor_maps.append(anchor_map)
        else:
            assert anchors is None

        h, w = math.ceil(h / 2), math.ceil(w / 2)

    anchor_maps = torch.cat(anchor_maps, 1).t()

    return anchor_maps


def build_anchor_map(image_size, map_size, anchor):
    cell_size = (image_size[0] / map_size[0], image_size[1] / map_size[1])

    y = torch.linspace(cell_size[0] / 2, image_size[0] - cell_size[0] / 2, map_size[0])
    x = torch.linspace(cell_size[1] / 2, image_size[1] - cell_size[1] / 2, map_size[1])

    y, x = torch.meshgrid(y, x)
    h = torch.ones(map_size) * anchor[0]
    w = torch.ones(map_size) * anchor[1]
    anchor_map = torch.stack([y, x, h, w])
    anchor_map = anchor_map.view(anchor_map.size(0), anchor_map.size(1) * anchor_map.size(2))

    return anchor_map
