import torch
    
# Domain-bridging features generation
def FeatStyle(x, alpha=0.1, eps=1e-6):
    beta = torch.distributions.Beta(alpha, alpha)
    B = x.size(0)
    mu = x.mean(dim=1, keepdim=True)  # (B, 1, C)
    var = x.var(dim=1, keepdim=True)  # (B, 1, C)

    sig = (var + eps).sqrt()
    mu, sig = mu.detach(), sig.detach()
    x_normed = (x - mu) / sig

    lmda = beta.sample((B, 1, 1))
    lmda = lmda.to(x.device)

    perm = torch.randperm(B)

    mu2, sig2 = mu[perm], sig[perm]
    mu_mix = mu * lmda + mu2 * (1 - lmda)
    sig_mix = sig * lmda + sig2 * (1 - lmda)

    return x_normed * sig_mix + mu_mix


def FeatureStylization(x, num_clips, alpha=0.1, eps=1e-6):

    x = x.view(-1, num_clips*x.size(-2), x.size(-1)) #[B,nt,C]
    B = x.size(0)
    # x.shape = (B, Seq, C)

    x_part1 = x[:B//2]
    x_part2 = x[B//2:]

    # x_part1 = FeatStyle(x_part1, alpha, eps)
    x_part2 = FeatStyle(x_part2, alpha, eps)

    x = torch.cat([x_part1, x_part2], dim=0)

    return x.view(B*num_clips, -1, x.size(-1))