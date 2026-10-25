import torch
import torch.nn.functional as F


def probe_accuracy(
    features,
    labels,
    train,
    test,
    n_classes,
    epochs=300,
    lr=1e-2,
    weight_decay=1e-3,
    seed=0,
):
    torch.manual_seed(seed)
    mean, std = features[train].mean(0), features[train].std(0) + 1e-6
    x = (features - mean) / std
    probe = torch.nn.Linear(x.size(1), n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        loss = F.cross_entropy(probe(x[train]), labels[train])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (probe(x[test]).argmax(-1) == labels[test]).float().mean().item()
