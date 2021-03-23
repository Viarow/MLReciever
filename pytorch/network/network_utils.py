import torch


def batch_matvec_mul(A, b, transpose_a=False):
    if transpose_a:
        C = torch.matmul(torch.transpose(A, 1, 2), torch.unsqueeze(b, dim=2))
    else:
        C = torch.matmul(A, torch.unsqueeze(b, dim=2))
    return torch.squeeze(C, dim=-1)


def batch_mat_trace(A):
    batch_trace = torch.sum(torch.diagonal(A, offset=0, dim1=-1, dim2=-2), dim=-1)
    return batch_trace