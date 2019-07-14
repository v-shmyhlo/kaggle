import torch


def pairwise_distances(input):
    delta = input.unsqueeze(0) - input.unsqueeze(1)
    dist = torch.norm(delta, 2, 2)

    return dist


def get_valid_positive_mask(labels):
    indices_equal = torch.eye(labels.size(0)).byte().to(labels.device)
    indices_not_equal = ~indices_equal

    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

    mask = indices_not_equal & label_equal
    return mask


def get_valid_negative_mask(labels):
    indices_equal = torch.eye(labels.size(0)).byte().to(labels.device)
    indices_not_equal = ~indices_equal

    label_not_equal = torch.ne(labels.unsqueeze(1), labels.unsqueeze(0))

    mask = indices_not_equal & label_not_equal
    return mask


def get_valid_triplets_mask(labels):
    """
    To be valid, a triplet (a,p,n) has to satisfy:
        - a,p,n are distinct embeddings
        - a and p have the same label, while a and n have different label
    """
    indices_equal = torch.eye(labels.size(0)).byte().to(labels.device)
    indices_not_equal = ~indices_equal
    i_ne_j = indices_not_equal.unsqueeze(2)
    i_ne_k = indices_not_equal.unsqueeze(1)
    j_ne_k = indices_not_equal.unsqueeze(0)
    distinct_indices = i_ne_j & i_ne_k & j_ne_k

    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
    i_eq_j = label_equal.unsqueeze(2)
    i_eq_k = label_equal.unsqueeze(1)
    i_ne_k = ~i_eq_k
    valid_labels = i_eq_j & i_ne_k

    mask = distinct_indices & valid_labels
    return mask


def batch_all_triplet_loss(input, target, margin):
    distances = pairwise_distances(input)

    anchor_positive_dist = distances.unsqueeze(2)
    anchor_negative_dist = distances.unsqueeze(1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # get a 3D mask to filter out invalid triplets
    mask = get_valid_triplets_mask(target)

    triplet_loss = triplet_loss * mask.float()
    triplet_loss.clamp_(min=0)

    # count the number of positive triplets
    epsilon = 1e-16
    num_positive_triplets = (triplet_loss > 0).float().sum()
    num_valid_triplets = mask.float().sum()
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + epsilon)

    triplet_loss = triplet_loss.sum() / (num_positive_triplets + epsilon)

    return triplet_loss  # , fraction_positive_triplets


def batch_hard_triplet_loss(input, target, margin):
    distances = pairwise_distances(input)

    mask_positive = get_valid_positive_mask(target)
    hardest_positive_dist = (distances * mask_positive.float()).max(dim=1)[0]

    mask_negative = get_valid_negative_mask(target)
    max_negative_dist = distances.max(dim=1, keepdim=True)[0]
    distances = distances + max_negative_dist * (~mask_negative).float()
    hardest_negative_dist = distances.min(dim=1)[0]

    triplet_loss = (hardest_positive_dist - hardest_negative_dist + margin).clamp(min=0)

    return triplet_loss
