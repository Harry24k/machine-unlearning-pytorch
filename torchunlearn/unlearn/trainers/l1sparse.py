import torch
import torch.nn as nn

from .unleaner import Unlearner


class L1Sparse(Unlearner):
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

    def __init__(self, rmodel, gamma=1e-5):
        super().__init__(rmodel)
        self.gamma = gamma
    
    def calculate_cost(self, train_data, reduction="mean"):
        r"""
        Overridden.
        """
        x, y = train_data['Retain']
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.rmodel(x)

        rt_loss = nn.CrossEntropyLoss(reduction="none")(logits, y)
        l1_loss = self.calculate_l1_penalty(self.rmodel)
        cost = rt_loss + self.gamma * l1_loss
        
        self.add_record_item("RTLoss", rt_loss.mean().item())
        self.add_record_item("L1loss", l1_loss.mean().item())
        self.add_record_item("Cost", cost.mean().item())

        return cost.mean() if reduction == "mean" else cost

    def calculate_l1_penalty(self, model):
        params_vec = []
        for param in model.parameters():
            params_vec.append(param.view(-1))
        return torch.linalg.norm(torch.cat(params_vec), ord=1)
        