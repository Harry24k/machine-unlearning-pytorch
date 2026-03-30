import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .unleaner import Unlearner
from ...attacks.attack import Attack

class ARU(Attack, Unlearner):
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
    
    def __init__(self, rmodel, margin=1.0, eps=0.05, steps=50, omit_label=1, batch_size=128, random_start = True):
        Unlearner.__init__(self, rmodel)
        Attack.__init__(self, 'PGD', rmodel)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.omit_label = omit_label
        self.targeted = True
        self.attack_mode = 'targeted'
        self.supported_mode = ["default", "targeted"]
        self.margin = margin
        self.eps = eps
        self.steps = steps
        self.alpha = eps/steps*2
        self.random_start = random_start
        self.batch_size = batch_size
        self.rlike_loader = None
        self.rlike_dataset = None
        
    def omit_target_label(self, inputs, labels):
        with torch.no_grad():
            logits = self.rmodel(inputs.to(self.device))
            logits[:, self.omit_label] = float('-inf') 
            return logits.argmax(dim=1)

    def modified_attack(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.omit_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        adv_images.requires_grad = True
        
        for _ in range(self.steps):
            noise = torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            ).to(self.device) #each step - our novel loss landscape smoothing process
            adv_images_noisy = (adv_images + noise).detach().requires_grad_(True)
            adv_images_noisy = torch.clamp(adv_images_noisy, min=-1, max=1)
            outputs = self.get_logits(adv_images_noisy)

            if self.targeted:
                cost = -loss(outputs, target_labels) 
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(
                cost, adv_images_noisy, retain_graph=False, create_graph=False
            )[0]

            grad.requires_grad=True
        
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()  
            adv_images.requires_grad = True

        return adv_images
        
    def generate_retain_like(self, train_data):
        rlike_x, rlike_y = [], []
        self._target_map_function = self.omit_target_label
    
        x_f, y_f = train_data['Forget']
        if not isinstance(x_f, torch.Tensor):
            x_f = torch.as_tensor(x_f)
        if not isinstance(y_f, torch.Tensor):
            y_f = torch.as_tensor(y_f)
    
        forget_ds = TensorDataset(x_f, y_f)
        forget_loader = DataLoader(forget_ds, batch_size=self.batch_size, shuffle=True)
    
        self.rmodel.eval()
        for x_forget, y_forget in forget_loader:
            x_forget = x_forget.to(self.device)
            y_forget = y_forget.to(self.device)
    
            adv_x = self.modified_attack(x_forget, y_forget)
            with torch.no_grad():
                logits = self.rmodel(adv_x)
                pred = logits.argmax(dim=1)
                mask = pred != self.omit_label
                rlike_x.append(adv_x[mask].detach().cpu())
                rlike_y.append(pred[mask].detach().cpu())
    
        if len(rlike_x) == 0:
            self.rlike_loader = None
            return
    
        rlike_x = torch.cat(rlike_x, dim=0)
        rlike_y = torch.cat(rlike_y, dim=0)
        rlike_dataset = TensorDataset(rlike_x, rlike_y)
        self.rlike_loader = DataLoader(rlike_dataset, batch_size=self.batch_size, shuffle=True)
        
    def calculate_cost(self, train_data, reduction="mean"):
        r"""
        Overridden.
        """
        x_forget, y_forget = train_data['Forget']
        x_forget = x_forget.to(self.device)
        y_forget = y_forget.to(self.device)
        self.x_forget = x_forget.detach().cpu()
        self.y_forget = y_forget.detach()
        self.logits_forget = self.rmodel(x_forget)

        if self.rlike_loader is None:
            self.generate_retain_like(train_data)
        if self.rlike_loader is None:
            raise RuntimeError("rlike_loader is empty.")

        for x_retain, y_retain in self.rlike_loader : 
            x_retain = x_retain.to(self.device)
            y_retain = y_retain.to(self.device)
            self.x_retain = x_retain.detach().cpu()
            self.logits_retain = self.rmodel(x_retain)

        self.anchor = self.logits_forget
        self.positive = self.logits_retain.detach().mean(dim=0, keepdim=True).repeat(self.anchor.size(0), 1)
        self.negative = self.logits_forget.detach().mean(dim=0, keepdim=True).repeat(self.anchor.size(0), 1)
        
        # 정규화
        self.anchor = F.normalize(self.anchor, dim=1)
        self.positive = F.normalize(self.positive, dim=1)
        self.negative = F.normalize(self.negative, dim=1)

        # Cost
        d_ap = 1 - (self.anchor * self.positive).sum(dim=1)
        d_an = 1 - (self.anchor * self.negative).sum(dim=1)
        self.sda_cost = F.relu(d_ap - d_an + self.margin).mean()
        # self.p_cost = d_ap.mean() #pos
        # self.n_cost = -d_an.mean() #neg
        self.ce_cost = F.cross_entropy(self.logits_retain, y_retain)
    
        # 최종 cost 구성
        self.cost = self.sda_cost + self.ce_cost 
        # self.cost = self.p_cost + self.ce_cost #positive-only
        # self.cost = self.n_cost + self.ce_cost #negative-only
        self.add_record_item("Cost", self.cost.mean().item())
    
        return self.cost.mean() if reduction == "mean" else self.cost
