import torch
from ..trainer import Trainer
from ...utils import get_subloader
from collections import OrderedDict

r"""
Base class for adversarial trainers.

Functions:
    self.record_rob : function for recording standard accuracy and robust accuracy against FGSM, PGD, and GN.

"""


class Unlearner(Trainer):
    def __init__(self, rmodel, device=None):
        super().__init__(rmodel, device)
        self.unlearn_dict = None

    def record_rob(
        self,
        loaders,
        eps=None,
        alpha=None,
        steps=None,
        std=None,
        record_cosine=None,
        n_limit=None,
    ):
        if (alpha is None) and (steps is not None):
            raise ValueError("Both alpha and steps should be given for PGD.")
        elif (alpha is not None) and (steps is None):
            raise ValueError("Both alpha and steps should be given for PGD.")

        self.unlearn_dict = OrderedDict()
        self.unlearn_dict['loaders'] = {}
        for key in loaders.keys():
            self.unlearn_dict['loaders'][key] = get_subloader(loaders[key], n_limit)
        self.unlearn_dict["eps"] = eps
        self.unlearn_dict["alpha"] = alpha
        self.unlearn_dict["steps"] = steps
        self.unlearn_dict["std"] = std
        self.record_cosine = record_cosine
            

    def record_during_eval(self):
        if self.unlearn_dict is not None:            
            for flag, loader in self.unlearn_dict["loaders"].items():
                self.dict_record["Clean" + flag] = self.rmodel.eval_accuracy(loader)
    
                eps = self.unlearn_dict.get("eps")
                if eps is not None:
                    self.dict_record["FGSM" + flag] = self.rmodel.eval_rob_accuracy_fgsm(
                        loader, eps=eps, verbose=False
                    )
                    steps = self.unlearn_dict.get("steps")
                    alpha = self.unlearn_dict.get("alpha")
                    if steps is not None:
                        self.dict_record["PGD" + flag] = self.rmodel.eval_rob_accuracy_pgd(
                            loader, eps=eps, alpha=alpha, steps=steps, verbose=False
                        )
    
                std = self.unlearn_dict.get("std")
                if std is not None:
                    self.dict_record["GN" + flag] = self.rmodel.eval_rob_accuracy_gn(
                        loader, std=std, verbose=False
                    )

        if self.record_cosine:
            retain_loader = self.unlearn_dict['loaders']['(R)']
            forget_loader = self.unlearn_dict['loaders']['(F)']
            for x, y in retain_loader:
                retain_grad, retain_layer_grads = self.get_grad(x, y)
                break
            for x, y in forget_loader:
                forget_grad, forget_layer_grads = self.get_grad(x, y)
                break

            cosine_sim_fn = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
            self.dict_record["Cosine"] = cosine_sim_fn(retain_grad, forget_grad).item()

            layer_cosines = []
            for r_grad, f_grad in zip(retain_layer_grads, forget_layer_grads):
                # 각 layer gradient의 cosine similarity 계산
                cosine_val = cosine_sim_fn(r_grad, f_grad)
                layer_cosines.append(cosine_val.item())
            average_layer_cosine = sum(layer_cosines) / len(layer_cosines)
            self.dict_record["Cosine(Layer)"] = average_layer_cosine

    def get_grad(self, x, y):
        self.rmodel.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.rmodel(x)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        grads = torch.autograd.grad(loss, self.rmodel.parameters(), retain_graph=False, create_graph=False)
        
        grad_list = []
        per_layer_grads = []
        for grad in grads:
            if grad is not None:
                grad_flat = grad.contiguous().view(-1)
                grad_list.append(grad_flat)
                per_layer_grads.append(grad_flat)
        
        # 전체 gradient 벡터 생성
        grad_vector = torch.cat(grad_list)
        
        return grad_vector, per_layer_grads
        

        