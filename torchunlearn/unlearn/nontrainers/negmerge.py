from collections import OrderedDict
import logging
import copy
import torch
from tqdm import tqdm
import os

from ... import RobModel


class NegMerge:
    """
    NegMerge: Consensual Weight Negation for Strong Machine Unlearning (ICML 2025)

    Algorithm:
        1) Fine-tune pretrained model on forget data only → multiple candidates
        2) Compute forget task vectors: τ_i = θ_ft_i − θ_pre
        3) Sign-consensus merge: keep coordinates where candidates agree on sign
        4) Negate and apply: θ_unlearn = θ_pre − λ · τ_merged
    """

    def __init__(self, rmodel, device=None):
        assert isinstance(rmodel, RobModel)
        self.rmodel = rmodel
        if device is None:
            self.device = next(rmodel.parameters()).device
        else:
            self.device = device

    def fit(
        self,
        train_loaders,
        lrs=[1e-4, 5e-4, 1e-3],
        epochs=1,
        repeats=3,
        scaling=1.0,
        consensus_ratio=1.0,
        aggregation="mean",
        save_path=None,
        overwrite=False,
    ):
        """
        train_loaders must contain:
            train_loaders['Retain']  (used for evaluation only)
            train_loaders['Forget']  (used for fine-tuning + evaluation)

        Workflow (per the paper):
            1) For each LR, fine-tune K candidates on forget data only (standard CE)
            2) Compute task vectors: τ_i = θ_ft_i − θ_pre
            3) Sign-consensus merge → τ_merged
            4) Negate: θ_unlearn = θ_pre − scaling * τ_merged
            5) Select best config by (retain_acc − forget_acc)

        Args:
            lrs: list of learning rates to search over
            epochs: training epochs per candidate
            repeats: number of candidates per LR
            scaling: negation strength λ (paper's scaling hyperparameter)
            consensus_ratio: fraction of candidates that must agree on sign
            aggregation: 'mean', 'sum', or 'median' for combining task vectors
        """
        if save_path is not None:
            if os.path.exists(save_path):
                if overwrite:
                    logging.warning("Save file(s) will be overwritten: " + save_path)
                else:
                    raise ValueError("[%s] is already exists." % (save_path))
            else:
                os.makedirs(save_path)

        base_rmodel = copy.deepcopy(self.rmodel)

        best_value = -1e10
        best_params = None
        best_config = None

        for lr in lrs:
            candidate_models = []

            for i in range(repeats):
                rmodel = copy.deepcopy(base_rmodel)
                self._forget_finetune(
                    rmodel=rmodel,
                    forget_loader=train_loaders['Forget'],
                    lr=lr,
                    epochs=epochs,
                )

                candidate_models.append(copy.deepcopy(rmodel))

                if save_path is not None:
                    rmodel.save_dict(save_path + "/candidate_lr(%s)_repeat(%d).pth" % (str(lr), i))

                del rmodel

            merged_rmodel = self._negate_merge(
                base_rmodel=base_rmodel,
                candidate_models=candidate_models,
                scaling=scaling,
                consensus_ratio=consensus_ratio,
                aggregation=aggregation,
            )

            retain_accuracy = merged_rmodel.eval_accuracy(data_loader=train_loaders['Retain'])
            forget_accuracy = merged_rmodel.eval_accuracy(data_loader=train_loaders['Forget'])

            print(
                "- LR:", str(lr),
                "Retain (%):", retain_accuracy,
                ", Forget (%):", forget_accuracy
            )

            curr_value = retain_accuracy - forget_accuracy
            if best_value < curr_value:
                best_value = curr_value
                best_config = lr
                best_params = copy.deepcopy(dict(merged_rmodel.named_parameters()))

            if save_path is not None:
                merged_rmodel.save_dict(save_path + "/merged_lr(%s).pth" % str(lr))

            del merged_rmodel
            del candidate_models

        print("Best LR:", str(best_config))
        for name, p in self.rmodel.named_parameters():
            p.data = best_params[name].data
        del best_params

        if save_path is not None:
            self.rmodel.save_dict(save_path + "/best.pth")

    def _forget_finetune(
        self,
        rmodel,
        forget_loader,
        lr=1e-3,
        epochs=1,
    ):
        """
        Step 1 of NegMerge: standard fine-tuning on the forget set ONLY.

        This produces a model that has *learned* the forget data well.
        The task vector (θ_ft − θ_pre) then captures the direction of
        "knowing the forget data", which will be negated later.
        """
        rmodel.train()
        optimizer = torch.optim.SGD(rmodel.parameters(), lr=lr, momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for x_f, y_f in tqdm(forget_loader, leave=False):
                x_f, y_f = x_f.to(self.device), y_f.to(self.device)

                optimizer.zero_grad()
                out_f = rmodel(x_f)
                loss = loss_fn(out_f, y_f)
                loss.backward()
                optimizer.step()

    def _negate_merge(
        self,
        base_rmodel,
        candidate_models,
        scaling=1.0,
        consensus_ratio=1.0,
        aggregation="mean",
    ):
        """
        Steps 2-4 of NegMerge:
            2) τ_i = θ_ft_i − θ_pre          (forget task vectors)
            3) τ_merged = sign_consensus(τ_1, ..., τ_K)
            4) θ_unlearn = θ_pre − λ · τ_merged   (NEGATE)
        """
        if len(candidate_models) == 0:
            raise ValueError("candidate_models is empty.")

        base_params = OrderedDict(
            (name, p.clone().detach())
            for name, p in base_rmodel.named_parameters()
        )

        # Step 2: compute forget task vectors
        candidate_deltas = []
        for model in candidate_models:
            delta = OrderedDict()
            for name, p in model.named_parameters():
                delta[name] = p.clone().detach() - base_params[name]
            candidate_deltas.append(delta)

        # Step 3: sign-consensus merge
        merged_delta = self._sign_consensus_merge(
            candidate_deltas,
            consensus_ratio=consensus_ratio,
            aggregation=aggregation,
        )

        # Step 4: NEGATE — subtract the merged task vector
        merged_rmodel = copy.deepcopy(base_rmodel)
        for name, p in merged_rmodel.named_parameters():
            p.data = base_params[name] - scaling * merged_delta[name].to(p.device)

        return merged_rmodel

    def _sign_consensus_merge(
        self,
        candidate_deltas,
        consensus_ratio=1.0,
        aggregation="mean",
    ):
        """
        Sign-consensus merging (Algorithm 1 in the paper).

        For each parameter coordinate:
            - Count how many candidates have positive vs negative delta
            - Keep the coordinate only if sign agreement >= threshold
            - Aggregate surviving coordinates via mean/sum/median
            - Zero out non-consensus coordinates
        """
        if len(candidate_deltas) == 0:
            raise ValueError("candidate_deltas is empty.")

        merged = OrderedDict()
        n_models = len(candidate_deltas)
        threshold = max(1, int(n_models * consensus_ratio + 0.999999))

        param_names = list(candidate_deltas[0].keys())

        for name in param_names:
            stack = torch.stack([delta[name] for delta in candidate_deltas], dim=0)

            pos_count = (stack > 0).sum(dim=0)
            neg_count = (stack < 0).sum(dim=0)

            keep_pos = pos_count >= threshold
            keep_neg = neg_count >= threshold
            keep_mask = keep_pos | keep_neg

            if aggregation == "mean":
                agg = stack.mean(dim=0)
            elif aggregation == "sum":
                agg = stack.sum(dim=0)
            elif aggregation == "median":
                agg = stack.median(dim=0).values
            else:
                raise ValueError("aggregation must be one of ['mean', 'sum', 'median'].")

            merged_tensor = torch.zeros_like(agg)

            pos_tensor = torch.where(agg > 0, agg, torch.zeros_like(agg))
            neg_tensor = torch.where(agg < 0, agg, torch.zeros_like(agg))

            merged_tensor = torch.where(keep_pos, pos_tensor, merged_tensor)
            merged_tensor = torch.where(keep_neg, neg_tensor, merged_tensor)
            merged_tensor = torch.where(keep_mask, merged_tensor, torch.zeros_like(merged_tensor))

            merged[name] = merged_tensor

        return merged
