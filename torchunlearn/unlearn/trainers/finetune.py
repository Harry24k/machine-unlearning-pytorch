import torch
import torch.nn as nn

from .unleaner import Unlearner


class Finetune(Unlearner):
    r"""
    Attributes:
        self.rmodel : rmodel.
        self.device : device where rmodel is.
        self.optimizer : optimizer.
        self.scheduler : scheduler (Automatically updated).
        self.curr_epoch : current epoch starts from 1 (Automatically updated).
        self.curr_iter : current iters starts from 1 (Automatically updated).

    Arguments:
        rmodel (nn.Module): rmodel to train.
    """

    def __init__(self, rmodel):
        super().__init__(rmodel)
    
    def calculate_cost(self, train_data, reduction="mean"):
        r"""
        Overridden.
        """
        x, y = train_data['Retain']
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.rmodel(x)

        cost = nn.CrossEntropyLoss(reduction="none")(logits, y)
        self.add_record_item("Cost", cost.mean().item())

        return cost.mean() if reduction == "mean" else cost
