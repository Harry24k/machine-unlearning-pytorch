from collections import OrderedDict
import logging
import copy
import torch
from torch.autograd import grad
from tqdm import tqdm
import os


from ... import RobModel

class Influence:
    def __init__(self, rmodel, device=None):
        assert isinstance(rmodel, RobModel)
        self.rmodel = rmodel
        if device is None:
            self.device = next(rmodel.parameters()).device
        else:
            device = device
            
        self.N = 1000 # OTHER
        # self.N = 300000 # ImageNet

    def fit(self, train_loaders, alphas=[1e-9, 1e-8, 1e-7, 1e-6], repeat=1, save_path=None, overwrite=False):
        if os.path.exists(save_path):
            if overwrite:
                logging.warning("Save file(s) will be overwritten:" + save_path)
            else:
                raise ValueError("[%s] is already exists." % (save_path))
        else:
            os.makedirs(save_path)
        
        retain_loader = train_loaders["Retain"]
        forget_loader = train_loaders["Forget"]
        
        params = []
        for param in get_require_grad_params(self.rmodel, named=False):
            params.append(param.view(-1))
    
        forget_grad = torch.zeros_like(torch.cat(params)).to(self.device)
        retain_grad = torch.zeros_like(torch.cat(params)).to(self.device)
    
        total = 0
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.rmodel.eval()
        for data, label in tqdm(forget_loader):
            self.rmodel.zero_grad()
            real_num = data.shape[0]
            data = data.to(self.device)
            label = label.to(self.device)
            output = self.rmodel(data)

            loss = criterion(output, label)
            f_grad = sam_grad(self.rmodel, loss) * real_num
            forget_grad += f_grad
            total += real_num

        total_2 = 0
        for data, label in tqdm(retain_loader):
            self.rmodel.zero_grad()
            real_num = data.shape[0]
            data = data.to(self.device)
            label = label.to(self.device)
            output = self.rmodel(data)

            loss = criterion(output, label)
            r_grad = sam_grad(self.rmodel, loss) * real_num
            retain_grad += r_grad
            total_2 += real_num
    
        retain_grad *= total / ((total + total_2) * total_2)
        forget_grad /= total + total_2
    
        perturb = woodfisher(
            self.rmodel,
            retain_loader,
            criterion=criterion,
            v=forget_grad - retain_grad,
            N=self.N,
            device=self.device
        )

        best_value = -1e10
        for alpha in alphas:
            for i in range(repeat):
                rmodel = copy.deepcopy(self.rmodel)
        
                apply_perturb(rmodel, alpha * perturb)

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

def woodfisher(rmodel, loader, criterion, v, N, device):
    rmodel.eval()
    k_vec = torch.clone(v)
    o_vec = None
    n = 0
    for batch_data, batch_label in tqdm(loader):
        for idx in range(len(batch_data)):
            n += 1
            rmodel.zero_grad()
            data = batch_data[idx:idx+1].to(device)
            label = batch_label[idx:idx+1].to(device)
            output = rmodel(data)
    
            loss = criterion(output, label)
            sample_grad = sam_grad(rmodel, loss)
            with torch.no_grad():
                if o_vec is None:
                    o_vec = torch.clone(sample_grad)
                else:
                    tmp = torch.dot(o_vec, sample_grad)
                    k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                    o_vec -= (tmp / (N + tmp)) * o_vec
            if n > N:
                return k_vec
    return k_vec
        
def get_require_grad_params(rmodel, named=False):
    if named:
        return [
            (name, param)
            for name, param in rmodel.named_parameters()
            if param.requires_grad
        ]
    else:
        return [param for param in rmodel.parameters() if param.requires_grad]

def sam_grad(rmodel, loss):
    names = []
    params = []
    for param in get_require_grad_params(rmodel, named=False):
        params.append(param)

    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]

    return torch.cat(sample_grad)

def apply_perturb(rmodel, v):
    curr = 0
    for param in get_require_grad_params(rmodel, named=False):
        length = param.view(-1).shape[0]
        param.view(-1).data += v[curr : curr + length].data
        curr += length
    
    