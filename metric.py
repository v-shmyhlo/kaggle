def accuracy(input, target, topk=1):
    _, input = input.topk(topk, 1)
    target = target.unsqueeze(1)
    acc = (input == target).float().sum(1)

    return acc
