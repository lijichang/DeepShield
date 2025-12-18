import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # cos sim
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        # LogSumExp
        max_val, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = (similarity_matrix - max_val.detach()) / self.temperature

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, 
            torch.arange(batch_size).view(-1, 1).to(features.device), 0
        )
        
        mask = mask * logits_mask
        
        epsilon = 1e-8
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + epsilon)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()
        
        return loss