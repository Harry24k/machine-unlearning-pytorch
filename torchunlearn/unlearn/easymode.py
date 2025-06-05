from ..utils.data import UnlearnDataSetup, MergedLoaders
from copy import deepcopy

class Unlearn:
    def __init__(self, rmodel, pretrained_path, train_loaders, test_loaders, save_path, overwrite=False, copy=True, 
                 record_cosine=False, record_type="Epoch"):
        self.rmodel = rmodel
        if pretrained_path is not None:
            self.rmodel.load_dict(pretrained_path)
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.save_path = save_path
        self.overwrite = overwrite
        self.copy = copy

        loaders_with_flags = {
            "(R)": train_loaders['Retain'],
            "(F)": train_loaders['Forget'],
            "(Te)": test_loaders['Test'],
        }
        self.loaders_with_flags = loaders_with_flags
        self.record_cosine = record_cosine
        self.record_type = record_type
        self.save_best = {"Clean(R)":"HB", "Clean(F)":"LBO"}

    def get_rmodel(self):
        if self.copy:
            return deepcopy(self.rmodel)
        else:
            return self.rmodel

    def finetune(self, epoch, lr, momentum=0.9, weight_decay=5e-4, scheduler=None, n_validation=1e10):
        rmodel = self.get_rmodel()
        train_loaders = self.train_loaders 
        test_loaders = self.test_loaders
        
        from .trainers.finetune import Finetune
        trainer = Finetune(rmodel)
        merged_train_loader = MergedLoaders(train_loaders)
        trainer.setup(optimizer=f"SGD(lr={lr}, momentum={momentum}, weight_decay={weight_decay})",
                      scheduler=scheduler, scheduler_type=None,
                      minimizer=None, n_epochs=epoch,
                     )
        
        trainer.record_rob(self.loaders_with_flags, record_cosine=self.record_cosine, n_limit=n_validation)
        trainer.fit(train_loaders=merged_train_loader, n_epochs=epoch,
                    save_path=self.save_path, save_best=self.save_best,
                    save_type=None, save_overwrite=self.overwrite, record_type=self.record_type)
        return rmodel

    def neggrad(self, epoch, lr, momentum=0.9, weight_decay=5e-4, scheduler=None, n_validation=1e10):
        rmodel = self.get_rmodel()
        train_loaders = self.train_loaders 
        test_loaders = self.test_loaders
        
        from .trainers.neggrad import NegGrad
        trainer = NegGrad(rmodel)
        merged_train_loader = MergedLoaders(train_loaders)
        trainer.setup(optimizer=f"SGD(lr={lr}, momentum={momentum}, weight_decay={weight_decay})",
                      scheduler=scheduler, scheduler_type=None,
                      minimizer=None, n_epochs=epoch,
                     )
        
        trainer.record_rob(self.loaders_with_flags, record_cosine=self.record_cosine, n_limit=n_validation)
        trainer.fit(train_loaders=merged_train_loader, n_epochs=epoch,
                    save_path=self.save_path, save_best=self.save_best,
                    save_type=None, save_overwrite=self.overwrite, record_type=self.record_type)
        return rmodel
        
    def influence(self, alphas=[1, 10, 20, 30, 50, 100], repeat=3):
        rmodel = self.get_rmodel()
        train_loaders = self.train_loaders
        
        from .nontrainers.influence import Influence
        trainer = Influence(rmodel)
        trainer.fit(train_loaders, save_path=self.save_path, alphas=alphas, repeat=repeat, overwrite=self.overwrite)
        return rmodel

    def fisher_forget(self, alphas=[1e-9, 1e-8, 1e-7, 1e-6], repeat=3):
        rmodel = self.get_rmodel()
        train_loaders = self.train_loaders

        omit_label = self.check_omit_label()
        
        from .nontrainers.fisherforget import FisherForget
        trainer = FisherForget(rmodel)
        trainer.fit(train_loaders, save_path=self.save_path, alphas=alphas, repeat=repeat, omit_label=omit_label, overwrite=self.overwrite)
        return rmodel

    def check_omit_label(self, n_check_batches=5):
        unique_labels = set()
        omit_label = None
        for i, (_, y) in enumerate(self.train_loaders['Forget']):
            unique_labels.update(y.numpy())
            if i >= n_check_batches:  # Check until n batches
                break
            elif len(unique_labels) > 1:
                break
            else:
                omit_label = y.numpy()[0]
        return omit_label
        
        