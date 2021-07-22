import torch
import torch.nn.functional as f
import re
import numpy as np
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tqdm
        
class RefBasedDeepMetricTrainer(object):

    def __init__(self, 
        train_dls, 
        valid_dls,
        refs_list,
        batch_sizes,
        num_epoch,
        device,
        model,
        optimizer_f,
        optimizer_kwargs,
        scheduler_f,
        scheduler_kwargs,
        result_path,
        enab_summary_writer = True,
        enab_tqdm=True,
    ):
        
        self.train_dls = train_dls
        self.valid_dls = valid_dls
        self.refs_list = refs_list
        self.max_iter_per_seq = 800# limit for iterations in each Seq data.

        if isinstance(batch_sizes, int):
            self.batch_sizes = [batch_sizes] * 3
        elif isinstance(batch_sizes, list):
            self.batch_sizes = batch_sizes

        self.num_epoch = num_epoch

        self.device = device
        self.model = model.to(device)
        self.optimizer_f = optimizer_f
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_f = scheduler_f
        self.scheduler_kwargs = scheduler_kwargs

        self.result_path = result_path
        self.enab_tqdm = enab_tqdm

        self.train_it = 0
        self.train_ep = 0
        self.valid_it = 0
        self.valid_ep = 0

        self.num_train_it = 0
        for dl in self.train_dls:
            self.num_train_it += len(dl)

        self.num_valid_it = 0
        for dl in self.valid_dls:
            self.num_valid_it += len(dl)

        self._train_loss = np.zeros(self.num_train_it)
        self._valid_loss = np.zeros(self.num_valid_it)

        self._optimizer = self.optimizer_f(self.model.parameters(), **optimizer_kwargs)
        self._scheduler = self.scheduler_f(self._optimizer, self.num_train_it, self.num_epoch, **scheduler_kwargs)

        self.weight_path = os.path.join(result_path, 'weight')
        self.tensorboard_path = os.path.join(result_path, 'tensorboard')


        os.makedirs(self.weight_path , exist_ok=True)

        if enab_summary_writer:
            os.makedirs(self.tensorboard_path, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=self.tensorboard_path)
        else:
            self.summary_writer = None

        
    def run_train(self, epoch=None):
        print("Now Epoch:{} is training".format(epoch))
        self.model.train()

        for (refs, train_dl) in zip(self.refs_list,self.train_dls):
            # obtain the refs in every seq
            curr_train_dl = train_dl
  
            curr_refs = refs.float().to(self.device)

            # Training
            dl_iter = iter(curr_train_dl)

            # This two only used to compute the loss and acc in every Sequence
            running_loss = 0
            corrects = 0
            for idx in range(len(curr_train_dl)):
                if idx >= self.max_iter_per_seq:
                    break
                try:
                    img1, img2, targets = next(dl_iter)

                except (StopIteration, TypeError):
                    dl_iter = iter(curr_train_dl)
                    img1, img2, targets = next(dl_iter)

                self._scheduler.zero_grad()
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)

                targets = targets.view(-1)
                targets = targets.to(self.device)
                
                loss, preds = self.model.compute_loss(img1, img2, curr_refs, targets)
        
                self.write_log("train", targets, preds, loss)
                loss.backward()

                self._scheduler.step()
                self.train_it += 1

        self.write_per_epoch_log("train")
        self.train_ep += 1
        self.train_it = 0


    def run_valid(self, epoch=None):
        self.model.eval()
        with torch.no_grad():
            for (refs, valid_dl) in zip(self.refs_list,self.valid_dls):
                # obtain the refs in every seq
                curr_valid_dl = valid_dl
                
                curr_refs = refs.float().to(self.device)

                # Training
                dl_iter = iter(curr_valid_dl)

                # This two only used to compute the loss and acc in every Sequence
                running_loss = 0
                corrects = 0
                for idx in range(len(curr_valid_dl)):
                    if idx >= self.max_iter_per_seq:
                        break
                    try:
                        img1, img2, targets = next(dl_iter)

                    except (StopIteration, TypeError):
                        dl_iter = iter(curr_valid_dl)
                        img1, img2, targets = next(dl_iter)

                    self._scheduler.zero_grad()
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    
                    targets = targets.view(-1)
                    targets = targets.to(self.device)
                    loss, preds = self.model.compute_loss(img1, img2, curr_refs, targets)
                   
                    self.write_log("valid", targets, preds, loss)

                    self.valid_it += 1

            self.write_per_epoch_log("valid")
            self.valid_ep += 1
            self.valid_it = 0

        return self._valid_loss.mean()
    
    
    def write_log(self, 
        phase,
        y:torch, 
        y_tilde:torch,
        loss:torch,
    ):
        if self.summary_writer is None:
            return

        assert phase is not None
        phase = phase.lower()

        loss_val = loss.item()

        if phase.startswith("train"):
            it = self.train_it + self.train_ep *  self.num_train_it
            self._train_loss[self.train_it] = loss_val

        elif phase.startswith("valid"):
            it = self.valid_it + self.valid_ep *  self.num_valid_it
            self._valid_loss[self.valid_it] = loss_val

        elif phase.startswith("test"): 
            it = self.test_it

        if phase == "train":
            ### Optimizer / Scheduler related values
            self.summary_writer.add_scalars(
                "Hyperparameter/Optimizer/",
                {
                    "LR"    : self._scheduler.get_lr()[0],
                    "beta1" : self._scheduler.optimizer.param_groups[0]["betas"][0],
                    "beta2" : self._scheduler.optimizer.param_groups[0]["betas"][1]
                },
                it
            )

        self.summary_writer.add_scalars(
            "Loss/{}/".format(phase),
            {
                "TotalLoss" : loss_val
            }, 
            it
        )

    def write_per_epoch_log(self,phase):
        
        if self.summary_writer is None:
            return

        assert phase is not None
        phase = phase.lower()

        if phase.startswith("train"):
            ep = self.train_ep
            loss_vect = self._train_loss
        elif phase.startswith("valid"):
            ep = self.valid_ep
            loss_vect = self._valid_loss
        elif phase.startswith("test"): 
            ep = self.test_ep

        self.summary_writer.add_scalars(
            "LossPerEpoch/{}/".format(phase),
            {
                "MeanLoss" : loss_vect.mean(),
                "StdLoss" : loss_vect.std(),
                "MinLoss" : loss_vect.min(),
                "MaxLoss" : loss_vect.max(),
            }, 
            ep
        )
        
    def run(self):
        if self.enab_tqdm:
            epoch_iterator = tqdm.tqdm(range(self.num_epoch))
        else:
            epoch_iterator = range(self.num_epoch)

        for ep in epoch_iterator:
            self.run_train(ep)
            self.run_valid(ep)
            self.save_model(ep)
            
    def run_optuna(self):
        
        best_val = np.inf
        if self.enab_tqdm:
            epoch_iterator = tqdm.tqdm(range(self.num_epoch))
        else:
            epoch_iterator = range(self.num_epoch)

        for ep in epoch_iterator:
            self.run_train(ep)
            tmp_val = self.run_valid(ep)
            self.save_model(ep)
            
            if tmp_val < best_val:
                best_val = tmp_val
                
        return best_val

    def save_model(self,epoch):
        torch.save(
            self.model.state_dict(), 
            os.path.join(
                self.result_path, 
                "weight",
                "epoch_{:03d}.weight".format(epoch)
            )
        )

        
class RefBasedDeepMetricTrainerV2(RefBasedDeepMetricTrainer):
    def __init__(self, 
        train_dls, 
        valid_dls,
        refs_list,
        batch_sizes,
        num_epoch,
        device,
        model,
        optimizer_f,
        optimizer_kwargs,
        scheduler_f,
        scheduler_kwargs,
        result_path,
        enab_summary_writer = True,
        enab_tqdm=True,
        per_seq_valid = True
    ):
        super(RefBasedDeepMetricTrainerV2, self).__init__(
            train_dls, 
            valid_dls,
            refs_list,
            batch_sizes,
            num_epoch,
            device,
            model,
            optimizer_f,
            optimizer_kwargs,
            scheduler_f,
            scheduler_kwargs,
            result_path,
            enab_summary_writer = True,
            enab_tqdm=True,
        )
        max_iter_per_seq = 150
        self.num_train_iter_per_dl = [max_iter_per_seq if len(dl)>= max_iter_per_seq else len(dl) for dl in self.train_dls]
        self.num_train_iter_per_dl = np.array(self.num_train_iter_per_dl)
        self.train_dl_prob = self.num_train_iter_per_dl / sum(self.num_train_iter_per_dl)
        self.train_iters = [iter(dl) for dl in self.train_dls]
        
        self.num_valid_iter_per_dl = [len(dl) for dl in self.valid_dls]
        #self.num_valid_iter_per_dl = [int(number / 3) for number in self.num_train_iter_per_dl]
        self.num_valid_iter_per_dl = np.array(self.num_valid_iter_per_dl)
        self.valid_dl_prob = self.num_valid_iter_per_dl / sum(self.num_valid_iter_per_dl)
        self.valid_iters = [iter(dl) for dl in self.valid_dls]
        
        self.per_seq_valid = per_seq_valid
        print("num_train_iter_per_dl: ",self.num_train_iter_per_dl)
        print("num_valid_iter_per_dl: ",self.num_valid_iter_per_dl)
        
        
    def reset_valid_loss_per_dl(self):
        self.valid_loss_per_dl = [np.zeros(num_batches) for num_batches in self.num_valid_iter_per_dl]
        
    def ret_max_valid_loss_per_seq(self):
        mean_valid_loss_per_dl = [valid_loss.mean() for valid_loss in self.valid_loss_per_dl]
        
        return np.max(mean_valid_loss_per_dl)
    
    def run_train(self, epoch=None):
        print("Now Epoch:{} is training".format(epoch))
        self.model.train()
        
        for i in range(self.num_train_it):
            dl_idx = np.random.choice(np.arange(len(self.train_dls), dtype=np.int), p=self.train_dl_prob)
            dl_iter = self.train_iters[dl_idx]
            refs = self.refs_list[dl_idx]
            curr_refs = refs.float().to(self.device)
 
            try:
                img1, img2, targets = next(dl_iter)

            except (StopIteration, TypeError):
                dl_iter = iter(self.train_dls[dl_idx])
                self.train_iters[dl_idx] = dl_iter
                img1, img2, targets = next(dl_iter)

            self._scheduler.zero_grad()
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)

            targets = targets.view(-1)
            targets = targets.to(self.device)

            loss, preds = self.model.compute_loss(img1, img2, curr_refs, targets)

            self.write_log("train", targets, preds, loss)
            loss.backward()

            self._scheduler.step()
            self.train_it += 1

        self.write_per_epoch_log("train")
        self.train_ep += 1
        self.train_it = 0
    
    def run_valid(self, epoch=None):
        if self.per_seq_valid:
            return self.run_valid_per_seq(epoch=epoch)
            
        else:
            return self.run_valid_random(epoch=epoch)

    def run_valid_random(self, epoch=None):
      
        self.model.eval()
        
        with torch.no_grad():
            for i in range(self.num_valid_it):
                dl_idx = np.random.choice(np.arange(len(self.valid_dls), dtype=np.int), p=self.valid_dl_prob)
                dl_iter = self.valid_iters[dl_idx]
                refs = self.refs_list[dl_idx]
                curr_refs = refs.float().to(self.device)

                try:
                    img1, img2, targets = next(dl_iter)
                except (StopIteration, TypeError):
                    dl_iter = iter(self.valid_dls[dl_idx])
                    self.valid_iters[dl_idx] = dl_iter
                    img1, img2, targets = next(dl_iter)

                img1 = img1.to(self.device)
                img2 = img2.to(self.device)

                targets = targets.view(-1)
                targets = targets.to(self.device)
                loss, preds = self.model.compute_loss(img1, img2, curr_refs, targets)

                self.write_log("valid", targets, preds, loss)

                self.valid_it += 1

            self.write_per_epoch_log("valid")
            self.valid_ep += 1
            self.valid_it = 0

        return self._valid_loss.mean()
    
    def run_valid_per_seq(self, epoch=None):

        self.model.eval()
        self.reset_valid_loss_per_dl()

        with torch.no_grad():
            for dl_idx,(refs, valid_dl, max_num) in enumerate(zip(self.refs_list,self.valid_dls, self.num_valid_iter_per_dl)):
                # obtain the refs in every seq
                curr_refs = refs.float().to(self.device)
                # Training
                dl_iter = iter(valid_dl)
                iter_max_num = max_num
                
                for batch_idx in range(iter_max_num):
                    try:
                        img1, img2, targets = next(dl_iter)

                    except (StopIteration, TypeError):
                        dl_iter = iter(valid_dl)
                        img1, img2, targets = next(dl_iter)

                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    
                    targets = targets.view(-1)
                    targets = targets.to(self.device)
                    loss, preds = self.model.compute_loss(img1, img2, curr_refs, targets)
                   
                    self.write_log("valid", targets, preds, loss)
                    
                    self.valid_loss_per_dl[dl_idx][batch_idx] = loss.item()

                    self.valid_it += 1

            self.write_per_epoch_log("valid")
            self.valid_ep += 1
            self.valid_it = 0

        return self.ret_max_valid_loss_per_seq()
