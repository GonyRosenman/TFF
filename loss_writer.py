from torch.nn import MSELoss,L1Loss,BCELoss
from losses import Percept_Loss
import csv
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from itertools import zip_longest
from metrics import Metrics

class Writer():
    """
    main class to handle logging the results, both to tensorboard and to a local csv file
    """
    def __init__(self,sets,**kwargs):
        self.register_args(**kwargs)
        self.register_losses(**kwargs)
        self.create_score_folders()
        self.metrics = Metrics()
        self.sets = sets
        self.total_train_steps = 0
        self.eval_iter = 0
        self.subject_accuracy = {}
        self.tensorboard = SummaryWriter(log_dir=self.tensorboard_dir, comment=self.experiment_title)
        for set in sets:
            setattr(self,'total_{}_loss_values'.format(set),[])
            setattr(self,'total_{}_loss_history'.format(set),[])
        for name, loss_dict in self.losses.items():
            if loss_dict['is_active']:
                for set in sets:
                    setattr(self, '{}_{}_loss_values'.format(name,set),[])
                    setattr(self, '{}_{}_loss_history'.format(name,set),[])

    def create_score_folders(self):
        self.tensorboard_dir = Path(os.path.join(self.log_dir, self.experiment_title))
        self.csv_path = os.path.join(self.experiment_folder, 'history')
        os.makedirs(self.csv_path)
        if self.task == 'fine_tune':
            self.per_subject_predictions = os.path.join(self.experiment_folder, 'per_subject_predictions')
            os.makedirs(self.per_subject_predictions)

    def save_history_to_csv(self):
        rows = [getattr(self, x) for x in dir(self) if 'history' in x and isinstance(getattr(self, x), list)]
        column_names = tuple([x for x in dir(self) if 'history' in x and isinstance(getattr(self, x), list)])
        export_data = zip_longest(*rows, fillvalue='')
        with open(os.path.join(self.csv_path, 'full_scores.csv'), 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(column_names)
            wr.writerows(export_data)


    def loss_summary(self,lr):
        self.scalar_to_tensorboard('learning_rate',lr,self.total_train_steps)
        loss_d = self.append_total_to_losses()
        for name, loss_dict in loss_d.items():
            if loss_dict['is_active']:
                for set in self.sets:
                    title = name + '_' + set
                    values = getattr(self,title + '_loss_values')
                    if len(values) == 0:
                        continue
                    score = np.mean(values)
                    history = getattr(self,title + '_loss_history')
                    history.append(score)
                    print('{}: {}'.format(title,score))
                    setattr(self,title + '_loss_history',history)
                    self.scalar_to_tensorboard(title,score)

    def accuracy_summary(self,mid_epoch):
        pred_all_sets = {x:[] for x in self.sets}
        truth_all_sets = {x:[] for x in self.sets}
        metrics = {}
        for subj_name,subj_dict in self.subject_accuracy.items():
            subj_pred = subj_dict['score'].mean().item()
            subj_error = subj_dict['score'].std().item()
            subj_truth = subj_dict['truth'].item()
            subj_mode = subj_dict['mode']
            with open(os.path.join(self.per_subject_predictions,'iter_{}.txt'.format(self.eval_iter)),'a+') as f:
                f.write('subject:{} ({})\noutputs: {:.4f}\u00B1{:.4f}  -  truth: {}\n'.format(subj_name,subj_mode,subj_pred,subj_error,subj_truth))
            pred_all_sets[subj_mode].append(subj_pred)
            truth_all_sets[subj_mode].append(subj_truth)
        for (name,pred),(_,truth) in zip(pred_all_sets.items(),truth_all_sets.items()):
            if len(pred) == 0:
                continue
            if self.fine_tune_task == 'regression':
                metrics[name + '_MAE'] = self.metrics.MAE(truth,pred)
                metrics[name + '_MSE'] = self.metrics.MSE(truth,pred)
                metrics[name +'_NMSE'] = self.metrics.NMSE(truth,pred)
            else:
                metrics[name + '_Balanced_Accuracy'] = self.metrics.BAC(truth,[x>0.5 for x in pred])
                metrics[name + '_Regular_Accuracy'] = self.metrics.RAC(truth,[x>0.5 for x in pred])
                metrics[name + '_AUROC'] = self.metrics.AUROC(truth,pred)

        for name,value in metrics.items():
            self.scalar_to_tensorboard(name,value)
            if hasattr(self,name):
                l = getattr(self,name)
                l.append(value)
                setattr(self,name,l)
            else:
                setattr(self, name, [value])
            print('{}: {}'.format(name,value))
        self.eval_iter += 1
        if mid_epoch and len(self.subject_accuracy) > 0:
            self.subject_accuracy = {k: v for k, v in self.subject_accuracy.items() if v['mode'] == 'train'}
        else:
            self.subject_accuracy = {}


    def write_losses(self,final_loss_dict,set):
        for loss_name,loss_value in final_loss_dict.items():
            title = loss_name + '_' + set
            loss_values_list = getattr(self,title + '_loss_values')
            loss_values_list.append(loss_value)
            if set == 'train':
                loss_values_list = loss_values_list[-self.running_mean_size:]
            setattr(self,title + '_loss_values',loss_values_list)

    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs

    def register_losses(self,**kwargs):
        self.losses = {'intensity':
                           {'is_active':False,'criterion':L1Loss(),'thresholds':[0.9, 0.99],'factor':kwargs.get('intensity_factor')},
                       'perceptual':
                           {'is_active':False,'criterion': Percept_Loss(**kwargs),'factor':kwargs.get('perceptual_factor')},
                       'reconstruction':
                           {'is_active':False,'criterion':L1Loss(),'factor':kwargs.get('reconstruction_factor')},
                       'binary_classification':
                           {'is_active':False,'criterion':BCELoss(),'factor':1},
                       'regression':
                           {'is_active':False,'criterion':L1Loss(),'factor':1}}
        if 'reconstruction' in kwargs.get('task').lower():
            self.losses['intensity']['is_active'] = True
            self.losses['perceptual']['is_active'] = True
            self.losses['reconstruction']['is_active'] = True
        elif kwargs.get('task').lower() == 'fine_tune':
            if kwargs.get('fine_tune_task').lower() == 'regression':
                self.losses['regression']['is_active'] = True
            else:
                self.losses['binary_classification']['is_active'] = True

    def append_total_to_losses(self):
        loss_d = self.losses.copy()
        loss_d.update({'total': {'is_active': True}})
        return loss_d

    def scalar_to_tensorboard(self,tag,scalar,iter=None):
        if iter is None:
            iter = self.total_train_steps if 'train' in tag else self.eval_iter
        if self.tensorboard is not None:
            self.tensorboard.add_scalar(tag,scalar,iter)

