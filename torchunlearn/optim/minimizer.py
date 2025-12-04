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


class CosineAnnealingUpdater:
    def __init__(self, initial_value, n_epochs, eta_min=0, start_epoch=0):
        """
        Mimics CosineAnnealingLR scheduler behavior with a delayed start.
        
        Args:
            initial_value (float): The starting value (e.g., learning rate).
            n_epochs (int): Total number of epochs.
            eta_min (float): Minimum value at the end of the annealing process.
            start_epoch (int): Epoch at which annealing should start.
        """
        self.initial_value = initial_value
        self.n_epochs = n_epochs
        self.eta_min = eta_min
        self.start_epoch = start_epoch
        self.current_epoch = 0

    def step(self):
        """Update the value following the cosine annealing schedule after start_epoch."""
        if self.current_epoch < self.start_epoch:
            new_value = self.initial_value  # Keep initial value before start_epoch
        else:
            effective_epoch = self.current_epoch - self.start_epoch
            total_effective_epochs = self.n_epochs - self.start_epoch
            if total_effective_epochs > 1:
                cos_factor = (1 + math.cos(math.pi * effective_epoch / total_effective_epochs)) / 2
                new_value = self.eta_min + (self.initial_value - self.eta_min) * cos_factor
            else:
                new_value = self.eta_min  # If only one effective epoch, directly use eta_min
        
        self.current_epoch += 1  # Move to the next epoch
        return new_value

    def reset(self):
        """Reset the scheduler to the first step."""
        self.current_epoch = 0

def reverse_proj_v1_on_v2(v1, v2, gamma=1.):
    shape = v1.shape
    v1 = v1.flatten(start_dim=0)
    v2 = v2.flatten(start_dim=0)
    norm_v2 = torch.dot(v2, v2)
    if norm_v2 == 0:
        return v1.reshape(shape)

    alpha = torch.dot(v1, v2)
    if alpha < 0:
        proj = -(alpha / norm_v2) * v2
        v1 = v1 + (1+gamma)*proj
    v1 = v1.reshape(shape)
    return v1

class UAM(Minimizer):
    def __init__(self, model, optimizer, rho, 
                 cosine_total_step=None, cosine_start_step=0, 
                 gamma=1.0, ascent_sample=None):
        super().__init__(model, optimizer)
        self.rho = rho
        if cosine_total_step is not None:
            self.rho_scheduler = CosineAnnealingUpdater(initial_value=rho, n_epochs=cosine_total_step, eta_min=0.0, start_epoch=cosine_start_step)
        else:
            self.rho_scheduler = None
        self.ascent_sample = ascent_sample
        self.gamma = gamma

    def step(self, cost_fn, inputs):
        if self.rho_scheduler is not None:
            self.rho = self.rho_scheduler.step()

        # Ascent step
        x_forget, y_forget = inputs['Forget']
        x_retain, y_retain = inputs['Retain']
        if self.ascent_sample is not None:
            x_forget, y_forget = x_forget[:self.ascent_sample], y_forget[:self.ascent_sample]
        x, y = torch.cat([x_forget, x_retain]), torch.cat([y_forget, y_retain])
        cost = cost_fn((x, y), reduction=None)
        cost = cost[:len(x_forget)].mean()  # Only consider the forget samples.
        cost.backward()
        self.ascent_step()

        # Descent step
        cost = cost_fn(inputs['Retain']) 
        cost.backward()
        self.descent_step()

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
