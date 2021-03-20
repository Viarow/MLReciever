import torch

def batch_matvec_mul(A, b, transpose_a=False):
    C = torch.matmul(A.transpose_(1, 2), b.unsqueeze_(dim=2))
    return C.squeeze_(dim=-1)