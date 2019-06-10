import torch


def collate_fn(batch):
    batch = list(zip(*batch))

    if len(batch) == 3:
        images, labels, ids = batch
        labels = torch.tensor(labels).float()
        rest = (labels,)
    else:
        images, ids = batch
        rest = ()

    images_tensor = torch.zeros(len(images), *images[0].size()[:-1], max(image.size(-1) for image in images))
    images_tensor.fill_(0.)

    for i, image in enumerate(images):
        images_tensor[i, ..., :image.size(-1)] = image

    return (images_tensor, *rest, ids)
