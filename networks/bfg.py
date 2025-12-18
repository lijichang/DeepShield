import torch

# boundary-expanded feature generation
def bfg(x, num_clips, expand_factor=1.1, eps=1e-6):
    """
    Scale the feature distribution outward or inward
    Args:
        x: input tensor [B*N,T,C]
        num_clips: number of clips to process
        expand_factor: factor to expand the distribution (>1 expands, <1 contracts)
        eps: small constant for numerical stability
    Returns:
        Scaled tensor with same shape as input
    """
    x = x.view(-1, num_clips*x.size(-2), x.size(-1)) #[B,nt,C]
    B = x.size(0)
    x_part1 = x[:B//2] #real
    x_part2 = x[B//2:] #fake

    # Calculate statistics
    mu = x_part2.mean(dim=1, keepdim=True)  # [B,1,C]
    var = x_part2.var(dim=1, keepdim=True)  # [B,1,C]
    sig = (var + eps).sqrt()
    
    # Normalize and scale
    x_normed = (x_part2 - mu) / sig
    x_scaled = x_normed * sig * expand_factor + mu

    x_scaled = torch.cat([x_part1, x_scaled], dim=0)
    
    return x_scaled.view(x.size(0)*num_clips, -1, x.size(-1))