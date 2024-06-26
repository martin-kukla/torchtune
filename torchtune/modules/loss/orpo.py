# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ORPOLoss(nn.Module):
    """
    ORPO Loss module: https://arxiv.org/abs/2403.07691

    Args:
        beta (float): Temperature parameter for the DPO loss, typically in the range of 0.1 to 0.5. Default is 0.1.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
        loss_type (str): Type of loss function to be used. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair'].
    """

    def __init__(
        self,
        beta: float = 0.1, # TODO: rename to lambda
        label_smoothing: float = 0.0, # TODO: support?
        loss_type: str = "sigmoid", # TODO: remove?
    ):
        super(ORPOLoss, self).__init__()
        self.beta = beta

    def forward(
        self,
        avg_chosen_logps: torch.Tensor,
        avg_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the ORPO loss for a batch of policy log probabilities. 
        TODO: Remove a batch of reference model log probabilities as parameter.

        Args:
            avg_chosen_logps (torch.Tensor): Average log probabilities
                for the chosen responses. Shape: (batch_size)
            avg_rejected_logps (torch.Tensor): Average log probabilities
                for the rejected responses. Shape: (batch_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors:
                - losses: The ORPO loss for each example in the batch.
                - chosen_rewards: Rewards for the chosen responses.
                - rejected_rewards: Rewards for the rejected responses.
                - nll_loss_chosen: NLL loss for chosen response.
                - or_loss: Odd ratio loss.
                - log_odds_chosen: Log odds chosen.
                - log_odds_rejected: Log odds rejected.

        Raises:
            ValueError: If an unknown loss type is specified.
        """
        # TODO: check maths - I wrote it in few mins
        log_odds_chosen = avg_chosen_logps - torch.log1p(-torch.exp(avg_chosen_logps))
        log_odds_rejected = avg_rejected_logps - torch.log1p(-torch.exp(avg_rejected_logps))
        or_loss = - F.logsigmoid(log_odds_chosen - log_odds_rejected)
        nll_loss_chosen = - avg_chosen_logps # Other implementations make extra call to CrossEntropy/NLLLoss pytorch layer, which I don't think we need?
        losses =  nll_loss_chosen + self.beta * or_loss

        chosen_rewards = (
            avg_chosen_logps.detach()
        )
        rejected_rewards = (
            avg_rejected_logps.detach()
        )

        return losses, chosen_rewards, rejected_rewards, nll_loss_chosen, or_loss, log_odds_chosen, log_odds_rejected
