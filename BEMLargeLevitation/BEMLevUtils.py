import torch

def get_indexes_subsample(N, centres):
    N = 400
    indexes = torch.zeros(centres.shape[2]).to(bool)
    mask = torch.randperm(centres.shape[2])[:N]
    indexes[mask] = True

    return indexes