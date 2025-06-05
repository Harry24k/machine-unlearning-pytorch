from collections import OrderedDict
import logging
import copy
import torch
from torch.autograd import grad
from tqdm import tqdm
import os

from ... import RobModel

class FisherForget:
    def __init__(self, rmodel, device=None):
        assert isinstance(rmodel, RobModel)
        self.rmodel = rmodel
        if device is None:
            self.device = next(rmodel.parameters()).device
        else:
            device = device

    def fit(self, train_loaders, alphas=[1e-9, 1e-8, 1e-7, 1e-6], repeat=1, omit_label=None, save_path=None, overwrite=False):
        if os.path.exists(save_path):
            if overwrite:
                logging.warning("Save file(s) will be overwritten:" + save_path)
            else:
                raise ValueError("[%s] is already exists." % (save_path))
        else:
            os.makedirs(save_path)
                
        hessian = self.get_hessian(train_loaders['Retain'])
        best_value = -1e10
        for alpha in alphas:
            for i in range(repeat):
                rmodel = copy.deepcopy(self.rmodel)
                for name, p in rmodel.named_parameters():
                    mu, var = self.get_mean_var(hessian[name], omit_label, alpha)
                    p.data = mu + var.sqrt() * torch.empty_like(p.data).normal_()
                    
                if save_path is not None:
                    rmodel.save_dict(save_path + "/alpha(%s)_repeat(%d).pth"%(str(alpha), i))
                    
                retain_accuracy = rmodel.eval_accuracy(data_loader=train_loaders['Retain'])
                forget_accuracy = rmodel.eval_accuracy(data_loader=train_loaders['Forget'])
                print("- Alpha:", str(alpha), "with Repeat #", str(i), 
                      "Retain (%):", retain_accuracy, ", Forget (%):", forget_accuracy)
                curr_value = retain_accuracy - forget_accuracy
                if best_value < curr_value:  # Higher is better (i.e., large gap between retain and forget)
                    best_params = copy.deepcopy(dict(rmodel.named_parameters()))
                    best_alpha = alpha
                    best_value = curr_value
                del rmodel

        print("Best Alpha:", str(best_alpha))
        for name, p in self.rmodel.named_parameters():
            p.data = best_params[name].data
        del best_params

        if save_path is not None:
            self.rmodel.save_dict(save_path + "/best.pth")
    
    def get_hessian(self, loaders):
        rmodel = copy.deepcopy(self.rmodel).eval()
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        hessian = OrderedDict()
    
        for name, p in rmodel.named_parameters():
            hessian[name] = p.clone().detach()
            hessian[name].grad2_acc = 0
    
        for data, orig_target in tqdm(loaders):
            data, orig_target = data.to(self.device), orig_target.to(self.device)
            output = rmodel(data)
            prob = torch.nn.functional.softmax(output, dim=-1).data
    
            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = loss_fn(output, target)
                rmodel.zero_grad()
                loss.backward(retain_graph=True)
                for name, p in rmodel.named_parameters():
                    if p.requires_grad:
                        hessian[name].grad2_acc += torch.mean(prob[:, y]) * p.grad.data.pow(2)
    
        for name, p in rmodel.named_parameters():
            hessian[name].grad2_acc /= len(loaders)

        del rmodel
        
        return hessian

    def get_mean_var(self, p, omit_label, alpha):
        var = copy.deepcopy(1.0 / (p.grad2_acc + 1e-8))
        var = var.clamp(max=1e3)
        if p.shape[0] == self.rmodel.n_classes:
            var = var.clamp(max=1e2)
        var = alpha * var
        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        mu = copy.deepcopy(p.clone().detach())
    
        if p.shape[0] == self.rmodel.n_classes:
            # Last layer
            if omit_label is not None:
                mu[omit_label] = 0
                var[omit_label] = 0.0001
            var *= 10
        elif p.ndim == 1:
            # BatchNorm
            var *= 10
        return mu, var
