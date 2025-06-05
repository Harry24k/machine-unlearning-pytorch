from collections import defaultdict
import math
import torch


class Minimizer(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.state = defaultdict(dict)
        self.records = {}

    def update_records(self, key, value):
        arr = self.records.get(key)
        if arr is None:
            self.records[key] = []
            arr = self.records.get(key)
        arr.append(value)

    def state_dict(self):
        """Returns the state of the minimizer as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["model", "optimizer", "state"]
        }

    def load_state_dict(self, state_dict):
        """Loads the minimizers state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        raise NotImplementedError


class SAM(Minimizer):
    def __init__(self, model, optimizer, rho):
        super().__init__(model, optimizer)
        self.rho = rho

    def step(self, cost_fn, *inputs):
        cost = cost_fn(*inputs)
        cost.backward()
        self.ascent_step()

        self.update_records("loss_0", cost.item())

        cost = cost_fn(*inputs)
        cost.backward()
        self.descent_step()

        self.update_records("loss_p", cost.item())

    @torch.no_grad()
    def ascent_step(self):
        grad_norm = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grad_norm.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grad_norm), p=2) + 1.0e-16

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            self.state[p]["eps"] = self.rho / grad_norm * p.grad.clone().detach()
            p.add_(self.state[p]["eps"])
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()
