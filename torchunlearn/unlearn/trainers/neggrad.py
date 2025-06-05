import torch
import torch.nn as nn

from .unleaner import Unlearner


class NegGrad(Unlearner):
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

    def __init__(self, rmodel, retain_lambda=0.0, l1_penalty_lambda=0.0):
        super().__init__(rmodel)
        self.retain_lambda = retain_lambda
        self.l1_penalty_lambda = l1_penalty_lambda

    def calculate_cost(self, train_data, reduction="mean"):
        r"""
        Overridden.
        """
        x_forget, y_forget = train_data['Forget']
        batch_size_forget = len(y_forget)
        
        if self.retain_lambda > 0:
            x_retain, y_retain = train_data['Retain']
            x = torch.cat([x_forget, x_retain])
            y = torch.cat([y_forget, y_retain])
        else:
            x = x_forget
            y = y_forget
        
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.rmodel(x)

        fg_loss = -1*nn.CrossEntropyLoss(reduction="none")(logits[:batch_size_forget], y[:batch_size_forget])
        rt_loss = 0
        if self.retain_lambda > 0:
            rt_loss = nn.CrossEntropyLoss(reduction="none")(logits[batch_size_forget:], y[batch_size_forget:])
            self.add_record_item("RTLoss", rt_loss.mean().item())
        l1_loss = self.calculate_l1_penalty(self.rmodel)
        cost = (1-self.retain_lambda)*fg_loss + self.retain_lambda*rt_loss + self.l1_penalty_lambda * l1_loss
        
        self.add_record_item("FGLoss", fg_loss.mean().item())
        self.add_record_item("L1Loss", l1_loss.mean().item())
        self.add_record_item("Cost", cost.mean().item())

        return cost.mean() if reduction == "mean" else cost

    def calculate_l1_penalty(self, model):
        params_vec = []
        for param in model.parameters():
            params_vec.append(param.view(-1))
        return torch.linalg.norm(torch.cat(params_vec), ord=1)
        
