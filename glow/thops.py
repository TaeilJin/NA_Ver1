import torch


def onehot(y, num_classes):
    y_onehot = torch.zeros(y.size(0), num_classes).to(y.device)
    if len(y.size()) == 1:
        y_onehot = y_onehot.scatter_(1, y.unsqueeze(-1), 1)
    elif len(y.size()) == 2:
        y_onehot = y_onehot.scatter_(1, y, 1)
    else:
        raise ValueError("[onehot]: y should be in shape [B], or [B, C]")
    return y_onehot


def sum(tensor, dim=None, keepdim=False):
    if dim is None:
        # sum up all dim
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


def mean(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross3":
        return tensor[:, 0::3, ...], tensor[:, 1::3, ...], tensor[:, 2::3, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def cat_feature(tensor_a, tensor_b):
    return torch.cat((tensor_a, tensor_b), dim=1)


def timesteps(tensor):
    return int(tensor.size(2))


# # The below code is adapted from github.com/juho-lee/set_transformer/blob/master/modules.py 
# class MAB(nn.Module):
#     def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
#         super(MAB, self).__init__()
#         self.dim_V = dim_V
#         self.num_heads = num_heads
#         self.fc_q = nn.Linear(dim_Q, dim_V)
#         self.fc_k = nn.Linear(dim_K, dim_V)
#         self.fc_v = nn.Linear(dim_K, dim_V)
#         if ln:
#             self.ln0 = nn.LayerNorm(dim_V)
#             self.ln1 = nn.LayerNorm(dim_V)
#         self.fc_o = nn.Linear(dim_V, dim_V)

#     def forward(self, Q, K):
#         Q = self.fc_q(Q)
#         K, V = self.fc_k(K), self.fc_v(K)

#         dim_split = self.dim_V // self.num_heads
#         Q_ = torch.cat(Q.split(dim_split, -1), 0)
#         K_ = torch.cat(K.split(dim_split, -1), 0)
#         V_ = torch.cat(V.split(dim_split, -1), 0)

#         A = torch.softmax(Q_.bmm(K_.transpose(0,1))/math.sqrt(self.dim_V), -1)
#         O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), -1)
#         O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
#         O = O + F.relu(self.fc_o(O))
#         #O = O + F.elu(self.fc_o(O))
#         O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
#         return O

# class SAB(nn.Module):
#     def __init__(self, dim_in, dim_out, num_heads, ln=False):
#         super(SAB, self).__init__()
#         self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

#     def forward(self, X):
#         return self.mab(X, X)