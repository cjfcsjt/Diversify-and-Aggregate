import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

from torch.nn.parallel import DistributedDataParallel as DDP
from base import BaseTrainer
from utils import MetricTracker, MetricTracker_scalars
from models.loss import WBCELoss,WBCELossOld, KDLoss, ACLoss, PKDLoss
from data_loader import ADE


class Trainer_base(BaseTrainer):
    """
    Trainer class for a base step
    """
    def __init__(
        self, model, optimizer, evaluator, config, task_info,
        data_loader, lr_scheduler=None, logger=None, gpu=None, visulized_dir=None
    ):
        super().__init__(config, logger, gpu)
        if not torch.cuda.is_available():
            logger.info("using CPU, this will be slow")
        elif config['multiprocessing_distributed']:
            if gpu is not None:
                torch.cuda.set_device(self.device)
                model.to(self.device)
                # When using a single GPU per process and per
                # DDP, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False

            else:
                model.to(self.device)
                # DDP will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = DDP(model)

        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = nn.DataParallel(model, device_ids=self.device_ids)

        self.optimizer = optimizer
        self.evaluator_val = evaluator[0]
        self.evaluator_test = evaluator[1]

        self.task_info = task_info
        self.n_old_classes = len(self.task_info['old_class'])  # 0
        self.n_new_classes = len(self.task_info['new_class'])  # 100-50: 100 | 100-10: 100 | 50-50: 50 |

        self.train_loader = data_loader[0]
        if self.train_loader is not None:
            self.len_epoch = len(self.train_loader)

        self.val_loader = data_loader[1]
        if self.val_loader is not None:
            self.do_validation = self.val_loader is not None

        self.test_loader = data_loader[2]
        if self.test_loader is not None:
            self.do_test = self.test_loader is not None

        self.lr_scheduler = lr_scheduler

        # For automatic mixed precision(AMP)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

        if self.evaluator_val is not None:
            self.metric_ftns_val = [getattr(self.evaluator_val, met) for met in config['metrics']]
        if self.evaluator_test is not None:
            self.metric_ftns_test = [getattr(self.evaluator_test, met) for met in config['metrics']]

        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce', 'loss_ac',
            writer=self.writer,
            colums=['total', 'counts', 'average'],
        )
        self.valid_metrics = MetricTracker_scalars(writer=self.writer)
        self.test_metrics = MetricTracker_scalars(writer=self.writer)

        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])
        pos_weight_new = torch.ones(
            [self.n_new_classes], device=self.device) * self.config['hyperparameter']['pos_weight_new']
        pos_weight_old = torch.ones(
            [self.n_old_classes], device=self.device) * self.config['hyperparameter']['pos_weight_old']
        self.BCELoss = WBCELoss(pos_weight=pos_weight_new, n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes)
        self.BCELossOld = WBCELossOld(
            pos_weight=pos_weight_old, n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes, reduction = 'none')
        self.ACLoss = ACLoss()

        self._print_train_info()
        ##### STAR
        self.visulized_dir = visulized_dir
        if not config['test']:
            self.compute_cls_number(self.config)
        ##### STAR
    def _print_train_info(self):
        self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
        self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce + {self.config['hyperparameter']['ac']} * L_ac")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
        else:
            self.model.freeze_bn(affine_freeze=False)

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(self.train_loader):
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                logit, features = self.model(data['image'], ret_intermediate=False)

                loss_mbce = self.BCELoss(
                    logit[:, -self.n_new_classes:],  # [N, |Ct|, H, W]
                    data['label'],                # [N, H, W]
                ).mean(dim=[0, 2, 3])  # [|Ct|]

                loss_ac = self.ACLoss(logit[:, 0:1]).mean(dim=[0, 2, 3])  # [1]

                loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() + self.config['hyperparameter']['ac'] * loss_ac.sum()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_mbce', loss_mbce.mean().item())
            self.train_metrics.update('loss_ac', loss_ac.mean().item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag

    def _valid_epoch(self, epoch):
        torch.distributed.barrier()
        
        log = {}
        self.evaluator_val.reset()
        self.logger.info(f"Number of val loader: {len(self.val_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, _ = self.model(data['image'])

                logit = torch.sigmoid(logit)
                pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]
                idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_val.add_batch(target, pred)

            if self.rank == 0:
                self.writer.set_step((epoch), 'valid')

            for met in self.metric_ftns_val:
                if len(met().keys()) > 2:
                    self.valid_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                else:
                    self.valid_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
                # if 'by_class' in met().keys():
                #     by_class_str = '\n'
                #     for i in range(len(met()['by_class'])):
                #         if i in self.evaluator_val.new_classes_idx:
                #             by_class_str = by_class_str + f"{i:2d} *{ADE[i]} {met()['by_class'][i]:.2f}\n"
                #         elif i in self.evaluator_val.old_classes_idx:
                #             by_class_str = by_class_str + f"{i:2d}  {ADE[i]} {met()['by_class'][i]:.2f}\n"
                #     log.update({met.__name__ + '_by_class': by_class_str})
        return log

    def _test(self, epoch=None):
        torch.distributed.barrier()

        log = {}
        self.evaluator_test.reset()
        self.logger.info(f"Number of test loader: {len(self.test_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, features = self.model(data['image'])
                logit = torch.sigmoid(logit)
                pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]

                idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]

                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_test.add_batch(target, pred)

            if epoch is not None:
                if self.rank == 0:
                    self.writer.set_step((epoch), 'test')

            for met in self.metric_ftns_test:
                if epoch is not None:
                    if len(met().keys()) > 2:
                        self.test_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                    else:
                        self.test_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
        return log


class Trainer_incremental(Trainer_base):
    """
    Trainer class for incremental steps
    """
    def __init__(
        self, model, model_old, optimizer, evaluator, config, task_info,
        data_loader, lr_scheduler=None, logger=None, gpu=None
    ):
        super().__init__(
            model=model, optimizer=optimizer, evaluator=evaluator, config=config, task_info=task_info,
            data_loader=data_loader, lr_scheduler=lr_scheduler, logger=logger, gpu=gpu)

        if config['multiprocessing_distributed']:
            if gpu is not None:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old, device_ids=[gpu])
            else:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old)
        else:
            if model_old is not None:
                self.model_old = nn.DataParallel(model_old, device_ids=self.device_ids)

        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce', 'loss_mbce_new_extra', 'loss_mbce_old_extra', 'loss_kd', 'loss_pkd', 'loss_dkd_pos', 'loss_dkd_neg', 'loss_ac',
            writer=self.writer, colums=['total', 'counts', 'average'],
        )
        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        self.KDLoss = KDLoss(pos_weight=None, reduction='none')
        self.PKDLoss = PKDLoss()
        ###### STAR
        prev_info_path = \
            str(config.save_dir)[:-len(str(config.save_dir).split('_')[-1])] + \
            str(config['data_loader']['args']['task']['step'] - 1) + \
            "/prototypes-epoch{}.pth".format(config['trainer']['epochs'])

        # prev_info = torch.load(prev_info_path)
        # self.prev_numbers = prev_info['numbers'].to(self.device)
        # self.prev_prototypes = prev_info['prototypes'].to(self.device)
        # self.prev_norm = prev_info['norm_mean_and_std'].to(self.device)
        # self.prev_noise = prev_info['noise'].to(self.device)
        ###### STAR

    def _print_train_info(self):
        self.logger.info(f"pos_weight_new - {self.config['hyperparameter']['pos_weight_new']}, pos_weight_old -{self.config['hyperparameter']['pos_weight_old']}")
        self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce + {self.config['hyperparameter']['mbce_new_extra']} * L_mbce_new_extra + {self.config['hyperparameter']['mbce_old_extra']} * L_mbce_old_extra + {self.config['hyperparameter']['pkd']} * L_pkd + {self.config['hyperparameter']['kd']} * L_kd "
                         f"+ {self.config['hyperparameter']['dkd_pos']} * L_dkd_pos + {self.config['hyperparameter']['dkd_neg']} * L_dkd_neg "
                         f"+ {self.config['hyperparameter']['ac']} * L_ac")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
            self.model.module.freeze_dropout()
        else:
            self.model.freeze_bn(affine_freeze=False)
            self.model.freeze_dropout()
        self.model_old.eval()

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                if self.model_old is not None:
                    with torch.no_grad():
                        logit_old, features_old = self.model_old(data['image'], ret_intermediate=True)
                        pred = logit_old.argmax(dim=1)  # pred: [N. H, W]
                        idx = (logit_old > 0.5).float()  # logit: [N, C, H, W]
                        idx = idx.sum(dim=1)  # logit: [N, H, W]
                        pred[idx == 0] = 0  # set background (non-target class)
                region_bg = torch.logical_and(data['label'] == 0, pred == 0) # N,H,W
                pseudo_label_region = torch.logical_and(data['label'] == 0, pred > 0) # N,H,W 
                logit, features = self.model(data['image'], ret_intermediate=True)
                
                ########## pixel from new step images
                ## new cls logits >--> all pixels [|Ct|]
                # WARNING: the memory bg should not been considered, since they may contain current step new classes
                loss_mbce = self.BCELoss(
                    logit[:, -self.n_new_classes:],  # [N, |Ct|, H, W]
                    data['label'],                # [N, H, W]
                ).mean(dim=[0, 2, 3])
                ############# new cls logits are welcome for more negative pixels ###############
                ## new cls logits >--> bg pixels (none-old none-new few-future)
                logit_bg = logit.permute(1, 0, 2, 3)[:, region_bg].unsqueeze(0).unsqueeze(3) # [1, |C0:t|, PixelNum, 1]
                # old logit on bg pixels has 0 labels
                logit_bg_label = torch.zeros(1, region_bg.shape[2], 1, requires_grad=False).to(self.device)
                mbce_new_bg = self.BCELoss(
                    logit_bg[:, -self.n_new_classes:],  # [N, |Ct|, H, W]
                    logit_bg_label,                # [N, H, W]
                )#.mean(dim=[0, 2, 3])  # [|Ct|] 
                
                ## new cls logits >--> old cls pixels (pseudo label grounded in fake bg pixels)
                logit_pseudo_old = logit.permute(1, 0, 2, 3)[:, pseudo_label_region].unsqueeze(0).unsqueeze(3) # N,H,W -> [1, |C0:t|, PixelNum, 1]
                # new cls logit on old cls pixels has 0 labels
                logit_pseudo_old_label =  torch.zeros(1, pseudo_label_region.shape[2], 1, requires_grad=False).to(self.device) # [N, H, W] -> # [N, 1, H, W] -> # [1, N, H, W] -> 1, PixelNum -> 1,1,PixelNum,1
                mbce_new_pseudo_old = self.BCELoss(
                    logit_pseudo_old[:,-self.n_new_classes:],  # [1, |C0:t|, PixelNum, 1]
                    logit_pseudo_old_label,                # [1,PixNum,1]
                )#.mean(dim=[0, 2, 3])
                mbce_new_extra = [mbce_new_bg, mbce_new_pseudo_old]
                loss_mbce_new_extra = torch.cat(mbce_new_extra, dim=2).mean(dim=[0, 2, 3])

                ############# old cls logits faces severe imbalance positive and negative pixels, if not careful update, may corrupt the original learned classification boundary###############
                ######## pos / neg distribution on step 0, step 1, .... are all different. adavantage is that later step gives new negatives, disadvantages is that later steps has different distribution (e.g., lack of labled positives ... )
                ## old cls logits (no bg) >--> foreground pixels
                region_fg = torch.logical_and(data['label'] > self.n_old_classes, data['label'] < 255) # N,H,W
                logit_fg = logit.permute(1, 0, 2, 3)[:, region_fg].unsqueeze(0).unsqueeze(3) # [1, |C0:t|, PixelNum e.g., 15-1 step 0: 2597086, 1]
                # logit_fg_label = data['label'].unsqueeze(1).permute(1, 0, 2, 3)[:, region_fg].unsqueeze(0).unsqueeze(3).squeeze(1) # [N, H, W] -> # [N, 1, H, W] -> # [1, N, H, W] -> 1, PixelNum -> 1,1,PixelNum,1
                # old logit on fg pixels has 0 labels
                logit_old_fg_label = torch.zeros(1, region_fg.shape[2], 1, requires_grad=False).to(self.device)
                mbce_old_fg = self.BCELossOld(
                    logit_fg[:,1:self.n_old_classes + 1],  # [1, |C0:t|, PixelNum, 1]
                    logit_old_fg_label,                # [1,PixNum,1]
                )  # [1, |C1:t|, PixelNum, 1]  .mean(dim=[0, 2, 3])

                ## old cls logits (no bg) >--> bg pixels (none-old none-new few-future)
                ## new cls logits >--> bg pixels (none-old none-new few-future)
                logit_bg = logit.permute(1, 0, 2, 3)[:, region_bg].unsqueeze(0).unsqueeze(3) # [1, |C0:t|, PixelNum, 1]
                # old logit on bg pixels has 0 labels
                logit_bg_label = torch.zeros(1, region_bg.shape[2], 1, requires_grad=False).to(self.device)
                mbce_old_bg = self.BCELossOld(
                    logit_bg[:, 1:self.n_old_classes + 1],  # [N, |Ct|, H, W]
                    logit_bg_label,                # [N, H, W]
                )  #.mean(dim=[0, 2, 3])  # [|Ct|] 
               
                ## old cls logits (no bg) >--> old class pixels (1. pseudo label grounded in fake bg pixels 2. mem fg), incremental step, this is also ok for memory
                # if logit_mem is not None:
                region_fg_mem = torch.logical_and(data['label'] > 0, data['label'] <= self.n_old_classes) # N,H,W 
                logit_fg_mem = logit.permute(1, 0, 2, 3)[:, region_fg_mem].unsqueeze(0).unsqueeze(3) # N,C,H,W -> [1, |C0:t|, PixelNum, 1]
                if logit_fg_mem.shape[2] != 0:
                    # old logit on mem fg pixels
                    logit_fg_mem_label = data['label'].unsqueeze(1).permute(1, 0, 2, 3)[:, region_fg_mem].unsqueeze(0).unsqueeze(3).squeeze(1) # [N, H, W] -> # [N, 1, H, W] -> # [1, N, H, W] -> 1, PixelNum -> 1,1,PixelNum,1
                    mbce_old_fg_mem = self.BCELossOld(
                        logit_fg_mem[:,1: self.n_old_classes + 1],  # [1, |C0:t|, PixelNum, 1]
                        logit_fg_mem_label,                # [1,PixNum,1]
                    )  # .mean(dim=[0, 2, 3])
                else:
                    mbce_old_fg_mem = torch.zeros((1,self.n_old_classes,1,1)).to(self.device)
                mbce_old_extra = [mbce_old_fg_mem, mbce_old_bg, mbce_old_fg] #, mbce_old_bg, mbce_old_pseudo_old]
                loss_mbce_old_extra = torch.cat(mbce_old_extra, dim=2).mean(dim=[0, 2, 3])
               
                # [|C0:t-1|]
                loss_kd = self.KDLoss(
                    logit[:, 1:self.n_old_classes + 1],  # [N, |C0:t|, H, W]
                    logit_old[:, 1:].sigmoid()       # [N, |C0:t|, H, W]
                ).mean(dim=[0, 2, 3])

                # # [1]
                # loss_ac = self.ACLoss(logit[:, 0:1]).mean(dim=[0, 2, 3])

                # # [|C0:t-1|]
                # loss_dkd_pos = self.KDLoss(
                #     features['pos_reg'][:, :self.n_old_classes],
                #     features_old['pos_reg'].sigmoid()
                # ).mean(dim=[0, 2, 3])

                # # [|C0:t-1|]
                # loss_dkd_neg = self.KDLoss(
                #     features['neg_reg'][:, :self.n_old_classes],
                #     features_old['neg_reg'].sigmoid()
                # ).mean(dim=[0, 2, 3])
                loss_pkd = self.PKDLoss(features, features_old, pseudo_label_region.unsqueeze(1).to(torch.float32))

                loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() + \
                    self.config['hyperparameter']['mbce_new_extra'] * loss_mbce_new_extra.sum() + \
                    self.config['hyperparameter']['mbce_old_extra'] * loss_mbce_old_extra.sum() + \
                    self.config['hyperparameter']['kd'] * loss_kd.sum() + \
                    self.config['hyperparameter']['pkd'] * loss_pkd.sum()
                    # self.config['hyperparameter']['dkd_pos'] * loss_dkd_pos.sum() + self.config['hyperparameter']['dkd_neg'] * loss_dkd_neg.sum() + \
                    # self.config['hyperparameter']['ac'] * loss_ac.sum()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_mbce', loss_mbce.mean().item())
            self.train_metrics.update('loss_mbce_new_extra', loss_mbce_new_extra.mean().item())
            self.train_metrics.update('loss_mbce_old_extra', loss_mbce_old_extra.mean().item())
            self.train_metrics.update('loss_kd', loss_kd.mean().item())
            self.train_metrics.update('loss_pkd', loss_pkd.mean().item())
            # self.train_metrics.update('loss_ac', loss_ac.mean().item())
            # self.train_metrics.update('loss_dkd_pos', loss_dkd_pos.mean().item())
            # self.train_metrics.update('loss_dkd_neg', loss_dkd_neg.mean().item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag
