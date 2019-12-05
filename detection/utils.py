import numpy as np
import torch
from PIL import ImageFont, Image, ImageDraw


def logit(input):
    return torch.log(input / (1 - input))


def draw_boxes(image, detections, class_names, line_width=2):
    colors = np.random.RandomState(42).uniform(51, 255, size=(len(class_names), 3)).round().astype(np.uint8)
    font = ImageFont.truetype('./imet/Droid+Sans+Mono+Awesome.ttf', size=14)

    class_ids, boxes, scores = detections
    scores = scores.sigmoid()  # TODO: fixme

    device = image.device
    image = image.permute(1, 2, 0).data.cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    for c, (t, l, b, r), s in zip(class_ids.data.cpu().numpy(), boxes.data.cpu().numpy(), scores.data.cpu().numpy()):
        color = tuple(colors[c])
        if len(class_names) > 1:
            text = '{}: {:.2f}'.format(class_names[c], s)
            size = draw.textsize(text, font=font)
            draw.rectangle(((l, t - size[1]), (l + size[0] + line_width * 2, t)), fill=color)
            draw.text((l + line_width, t - size[1]), text, font=font, fill=(0, 0, 0))
        draw.rectangle(((l, t), (r, b)), outline=color, width=line_width)

    image = torch.tensor(np.array(image) / 255).permute(2, 0, 1).to(device)

    return image
