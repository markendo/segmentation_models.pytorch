import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter

import numpy as np
import wandb
import json

class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', num_channels=1, verbose=True,
                 thresholds=None):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.num_channels = num_channels
        self.device = device
        self._to_device()
        self.thresholds = thresholds

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics or []:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items() if (k != 'iou_score' and k != 'iou_thresh_score')]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, valid_epoch=None, valid_loader=None, num_valid_per_epoch=None, max_score=None, classes=None, save_dir=None, valid_common_pathologies=None):
        self.on_epoch_start()
        print("THIS IS A TEST FOR DEPENDENCIES")
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {}
        # handle when there are no metrics
        for metric in self.metrics or []:
            # used during training and validation to find best thresholds
            if metric.__name__ == 'iou_score':
                metrics_meters[metric.__name__] = (np.zeros((len(self.thresholds), self.num_channels)), 
                                                   np.zeros((len(self.thresholds), self.num_channels)))
            # used during testing when best thresholds have been found                                       
            elif metric.__name__ == 'iou_thresh_score':
                metrics_meters[metric.__name__] = (np.zeros(self.num_channels), np.zeros(self.num_channels))
            else:
                if metric.task:
                    metrics_meters[metric.__name__ + '_' + metric.task] = AverageValueMeter()       
                else:
                    metrics_meters[metric.__name__] = AverageValueMeter()

        n_iter = 0
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics or []:
                    if metric_fn.__name__ == 'iou_score' or metric_fn.__name__ == 'iou_thresh_score':
                        metric_value = metric_fn(y_pred, y, metrics_meters[metric_fn.__name__], self.thresholds)
                        metrics_meters[metric_fn.__name__] = metric_value
                    else:
                        metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                        if metric_fn.task:
                            metrics_meters[metric_fn.__name__ + '_' + metric_fn.task].add(metric_value)
                        else:
                            metrics_meters[metric_fn.__name__].add(metric_value)
                if 'iou_score' in metrics_meters:
                    iou_metric = {'iou_score': metrics_meters['iou_score']}
                    logs.update(iou_metric)
                elif 'iou_thresh_score' in metrics_meters:
                    iou_metric = {'iou_thresh_score': metrics_meters['iou_thresh_score']}
                    logs.update(iou_metric)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items() if (k != 'iou_score' and k != 'iou_thresh_score')}
                logs.update(metrics_logs)

                n_iter += 1
                if valid_epoch:
                    if n_iter % (len(dataloader) // num_valid_per_epoch) == 0:
                        max_score = self.intermediate_valid_run(max_score, logs[self.loss.__name__], valid_epoch, valid_loader, classes, save_dir, valid_common_pathologies, logs)
                        logs.update({'max_score': float(max_score)})

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
        return logs

    def intermediate_valid_run(self, max_score, train_loss, valid_epoch, valid_loader, classes, save_dir, valid_common_pathologies, train_logs):
        iou_thresholds = np.arange(0, 1, 0.05).tolist()
        valid_logs = valid_epoch.run(valid_loader)
        valid_ious = np.divide(valid_logs['iou_score'][0], valid_logs['iou_score'][1])
        valid_max_ious = np.amax(valid_ious, axis=0)
        if valid_common_pathologies:
            common_pathologies_ious = np.zeros(len(valid_common_pathologies))
            for common_pathology_index, pathology in enumerate(valid_common_pathologies):
                max_iou_index = classes.index(pathology)
                common_pathologies_ious[common_pathology_index] = valid_max_ious[max_iou_index]
            common_pathologies_miou = np.mean(common_pathologies_ious)
            # to find best thresholding, must use train set since validation set doesn't have most pathologies present
            train_ious = np.divide(train_logs['iou_score'][0], train_logs['iou_score'][1])
            train_max_ious_index = np.argmax(train_ious, axis=0)
            best_thresholds = [iou_thresholds[num] for num in np.nditer(train_max_ious_index)]
        else:
            valid_max_ious_index = np.argmax(valid_ious, axis=0)
            best_thresholds = [self.thresholds[num] for num in np.nditer(valid_max_ious_index)]
        valid_miou = np.mean(valid_max_ious)
        # logging
        wandb_logs = {classes[i]: valid_max_ious[i] for i in range(len(classes))}
        wandb_logs.update({"train loss": train_loss,
                    "validation loss": valid_logs['distillation_loss'],
                    "validation miou score": valid_miou,})
        wandb.log(wandb_logs)

        if valid_common_pathologies:
            if max_score < common_pathologies_miou:
                max_score = common_pathologies_miou
                torch.save(self.model.state_dict(), save_dir / "distilled_model.pth")
                with open(save_dir / "distilled_thresholds.txt", "w") as threshold_file:
                    json.dump(best_thresholds, threshold_file)
                print(f'Model saved with performance of {common_pathologies_miou} on common pathologies!')
        else:
            if max_score < valid_miou:
                max_score = valid_miou
                torch.save(self.model.state_dict(), save_dir / "distilled_model.pth")
                with open(save_dir / "distilled_thresholds.txt", "w") as threshold_file:
                    json.dump(best_thresholds, threshold_file)
                    # print("Best thresholds:", best_thresholds)
                print(f'Model saved with performance of {valid_miou}!')
        self.on_epoch_start()
        return max_score


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, thresholds, device='cpu', num_channels=1, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            num_channels=num_channels,
            verbose=verbose,
            thresholds=thresholds,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, thresholds, device='cpu', num_channels=1, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            num_channels=num_channels,
            verbose=verbose,
            thresholds=thresholds,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
