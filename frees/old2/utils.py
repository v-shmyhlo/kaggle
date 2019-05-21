import torch


def collate_fn(batch, pad_value):
    batch = list(zip(*batch))

    if len(batch) == 3:
        images, labels, ids = batch
        labels = torch.tensor(labels).float()
        rest = (labels,)
    else:
        images, ids = batch
        rest = ()

    images_tensor = torch.zeros(
        len(images),
        1,
        images[0].size(1),
        max(image.size(2) for image in images))
    images_tensor.fill_(pad_value)

    for i, image in enumerate(images):
        images_tensor[i, :, :, :image.size(2)] = image

    return (images_tensor, *rest, ids)
