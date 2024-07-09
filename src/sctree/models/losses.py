"""
Loss functions for the reconstruction term of the ELBO.
"""
from typing import Literal

import torch
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial


def get_likelihood(output, likelihood: Literal["NB", "ZINB"]):
    px_scale, px_r, px_rate, px_dropout = output
    px_r = torch.exp(px_r)
    if likelihood == "NB":
        return NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
    elif likelihood == "ZINB":
        return ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale, zi_logits=px_dropout)
    else:
        raise NotImplementedError(f"{likelihood} is not implemented.")


def loss_nb(x, outputs, weights, likelihood: Literal["NB", "ZINB"]):
    loss = -torch.sum(torch.stack(
        [weight * get_likelihood(output=output, likelihood=likelihood).log_prob(x).sum(dim=-1) for weight, output in zip(weights, outputs)],
        dim=-1), dim=-1)
    return loss


def loss_nb_leafwise(x, outputs, weights, likelihood: Literal["NB", "ZINB"]):
    loss = -torch.stack(
        [weight * get_likelihood(output=output, likelihood=likelihood).log_prob(x).sum(dim=-1) for weight, output in zip(weights, outputs)],
        dim=-1)
    return loss

def loss_nb_leafwise_unweighted(x, outputs, weights, likelihood: Literal["NB", "ZINB"]):
    loss = -torch.stack(
        [get_likelihood(output=output, likelihood=likelihood).log_prob(x).sum(dim=-1) for weight, output in zip(weights, outputs)],
        dim=-1)
    return loss

