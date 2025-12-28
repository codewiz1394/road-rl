import torch

def load_torch_model(path: str):
    """
    Generic torch model loader for minimalRL-based policies.
    """
    return torch.load(path, map_location="cpu")
