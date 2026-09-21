def cls_acc(output, target, topk=1):
    _, pred = output.topk(topk, 1, True, True)
    
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
    
    acc = correct_k / target.shape[0]
    return acc.item()