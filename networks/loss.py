import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import batched_scatter, batched_index_select


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def calc_content_loss(input, target, norm=False):
    mse_loss = nn.MSELoss()
    if (norm == False):
        return mse_loss(input, target)
    else:
        return mse_loss(mean_variance_norm(input), mean_variance_norm(target))


def calc_style_loss(input, target):
    mse_loss = nn.MSELoss()
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(input_mean, target_mean) + \
           mse_loss(input_std, target_std)


def recon_criterion(target, output, index):
    base = target[:, index, :, :, :]
    loss = torch.mean(torch.abs(output - target))
    return loss


def weighted_recon_criterion(real_xs, fake_x):
    b, n, c, h, w = real_xs.size()
    # if similarity is not None:
    #     xs_c = torch.matmul(similarity.view(b, 1, n), real_xs.view(b, n, c * h * w))
    #     xs_c = xs_c.view(b, c, h, w)
    # else:
    xs_c = torch.mean(real_xs, dim=1, keepdim=False)
    loss = torch.mean(torch.abs(xs_c - fake_x))
    return loss


def nearest_recon_critertion(real_xs, fake_x):
    b, n, c, h, w = real_xs.size()
    losses = [torch.mean(torch.abs(real_xs[:, i, :, :, :] - fake_x)) for i in range(n)]
    loss = min(losses)
    return loss


def local_recon_criterion(real_xs, fake_x, index, s=8):
#     """
#     Local Reconstruction Loss
#     :param real_xs: real images (32*3*3*128*128)
#     :param fake_x: generated fake images (32*3*128*128)
#     :param similarity: alpha (32*3)
#     :param indice_base: the recorded positions of selected base local representations (32*M)
#     :param indice_refs: the recorded positions of the matched reference representations (32*2*M)
#     :param index: the index of base feature (1)
#     :param s: resize the feature map
#     :return:
#     """
    base = real_xs[:, index, :, :, :]  # (32*3*128*128)
#     # refs = torch.cat([real_xs[:, :index, :, :, :], real_xs[:, (index + 1):, :, :, :]], dim=1)  # (32*2*3*128*128)
#     # base_similarity = similarity[:, index]  # (32*1)
#     # ref_similarities = torch.cat([similarity[:, :index], similarity[:, (index + 1):]], dim=1)  # (32*2)

#     base = F.interpolate(base, size=s)  # (32*3*8*8)

#     fake_x = F.interpolate(fake_x, size=s)

    return recon_criterion(base, fake_x)


def ms_loss(fake_x1, fake_x2, sim1, sim2):
    lz = torch.mean(torch.abs(fake_x1 - fake_x2)) \
         / torch.mean(torch.abs(sim1 - sim2))
    eps = 1 * 1e-5
    loss_lz = 1 / (lz + eps)
    return loss_lz
