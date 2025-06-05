import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import numpy as np

def kl_divergence(p, q):
    """
    Compute the KL divergence between two distributions p and q.
    Both p and q are assumed to be log-probabilities.
    """
    p = F.softmax(p, dim=1)  # Convert logits to probabilities
    q = F.softmax(q, dim=1)  # Convert logits to probabilities
    log_p = torch.log(p + 1e-10)  # To avoid log(0) errors
    kl_div = F.kl_div(log_p, q, reduction='batchmean')  # Batch-wise average
    return kl_div

def wasserstein_distance_(p, q):
    """
    Compute the Wasserstein distance between two distributions p and q, which are vectors.
    Both p and q are assumed to be probability distributions (after softmax).
    p and q are of shape (batch_size, num_labels).
    """
    # Convert logits to probabilities (softmax)
    p = F.softmax(p, dim=1).cpu().detach().numpy()  # Convert logits to probabilities
    q = F.softmax(q, dim=1).cpu().detach().numpy()  # Convert logits to probabilities
    
    # Compute Wasserstein distance for each pair in the batch
    wasserstein_values = []
    
    batch_size, num_labels = p.shape
    for i in range(batch_size):
        # Sort the probabilities to match the expected input of scipy's wasserstein_distance
        sorted_p = np.sort(p[i])
        sorted_q = np.sort(q[i])
        
        # Calculate the Wasserstein distance for each pair of distributions
        distance = wasserstein_distance(sorted_p, sorted_q)
        wasserstein_values.append(distance)
    
    return torch.tensor(wasserstein_values, dtype=torch.float32).mean()

import torch
import pandas as pd
from collections import defaultdict

@torch.no_grad()
def get_logit_similarity(base_logits, logits):
    assert base_logits.keys() == logits.keys()

    res = {}
    
    for key in base_logits.keys():
        base_logit, base_labels = base_logits[key]
        logit, labels = logits[key]
    
        # KL Divergence
        kl_value = kl_divergence(base_logit, logit).item()
    
        # Wasserstein Distance
        wd_value = wasserstein_distance_(base_logit, logit).item()
    
        # L1, L2 Distance
        l1_value = (base_logit - logit).abs().mean().item()
        l2_value = ((base_logit - logit)**2).mean().item()

        res['KL'] = kl_value
        res['WD'] = wd_value
        res['L1'] = l1_value
        res['L2'] = l2_value
    
    return res