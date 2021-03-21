import torch

def batch_matvec_mul(A, b, transpose_a=False):
    if transpose_a:
        C = torch.matmul(torch.transpose(A, 1, 2), torch.unsqueeze(b, dim=2))
    else:
        C = torch.matmul(A, torch.unsqueeze(b, dim=2))
    return torch.squeeze(C, dim=-1)