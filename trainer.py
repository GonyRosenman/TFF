from loss_writer import Writer
from learning_rate import LrHandler
from data_preprocess_and_load.dataloaders import create_dataloaders
import torch
import warnings
from tqdm import tqdm
from model import Encoder_Transformer_Decoder,Encoder_Transformer_finetune,AutoEncoder
from losses import get_intense_voxels

class Trainer():
    """
    main class to handle training, validation and testing.
    note: the order of commands in the constructor is necessary
    """
    def __init__(self,sets,**kwargs):
        self.register_args(**kwargs)
        self.lr_handler = LrHandler(**kwargs)
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(sets, **kwargs)
        self.create_model()
        self.initialize_weights()
        self.create_optimizer()
        self.lr_handler.set_schedule(self.optimizer)
        self.writer = Writer(sets,**kwargs)
        self.sets = sets
        self.eval_iter = 0

        for name, loss_dict in self.writer.losses.items():
            if loss_dict['is_active']:
                print('using {} loss'.format(name))
                setattr(self, name + '_loss_func', loss_dict['criterion'])

    def create_optimizer(self):
        lr = self.lr_handler.base_lr
        params = self.model.parameters()
        weight_decay = self.kwargs.get('weight_decay')
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def initialize_weights(self):
        if self.loaded_model_weights_path is not None:
            state_dict = torch.load(self.loaded_model_weights_path)
            self.lr_handler.set_lr(state_dict['lr'])
            self.model.load_partial_state_dict(state_dict['model_state_dict'])
            self.model.loaded_model_weights_path = self.loaded_model_weights_path
            text = 'loaded model weights:\nmodel location - {}\nlast learning rate - {}\nvalidation loss - {}\n'.format(
                self.loaded_model_weights_path, state_dict['lr'],state_dict['loss_value'])
            if 'accuracy' in state_dict:
                text += 'validation accuracy - {}'.format(state_dict['accuracy'])
            print(text)

    def create_model(self):
        dim = self.train_loader.dataset.dataset.get_input_shape()
        if self.task.lower() == 'fine_tune':
            self.model = Encoder_Transformer_finetune(dim,**self.kwargs)
        elif self.task.lower() == 'autoencoder_reconstruction':
            self.model = AutoEncoder(dim,**self.kwargs)
        elif self.task.lower() == 'transformer_reconstruction':
            self.model = Encoder_Transformer_Decoder(dim,**self.kwargs)
        if self.cuda:
            self.model = self.model.cuda()

    def eval_epoch(self,set):
        loader = self.val_loader if set == 'val' else self.test_loader
        self.eval()
        with torch.no_grad():
            for input_dict in tqdm(loader, position=0, leave=True):
                loss_dict, _ = self.forward_pass(input_dict)
                self.writer.write_losses(loss_dict, set=set)


    def testing(self):
        self.eval_epoch('test')
        self.writer.loss_summary()
        self.writer.accuracy_summary(mid_epoch=False)
        for metric_name in dir(self.writer):
            if 'history' not in metric_name:
                continue
            metric_score = getattr(self.writer,metric_name)[-1]
            print('final test score - {} = {}'.format(metric_name,metric_score))


    def training(self):
        for epoch in range(self.nEpochs):
            self.train_epoch(epoch)
            self.eval_epoch('val')
            print('______epoch summary {}/{}_____\n'.format(epoch,self.nEpochs))
            self.writer.loss_summary()
            self.writer.accuracy_summary(mid_epoch=False)
            self.writer.save_history_to_csv()
            self.save_checkpoint_(epoch)


    def train_epoch(self,epoch):
        self.train()
        for batch_idx,input_dict in enumerate(tqdm(self.train_loader,position=0,leave=True)):
            self.writer.total_train_steps += 1
            self.optimizer.zero_grad()
            loss_dict, loss = self.forward_pass(input_dict)
            loss.backward()
            self.optimizer.step()
            self.lr_handler.schedule_check_and_update()
            self.writer.write_losses(loss_dict, set='train')
            if (batch_idx + 1) % self.validation_frequency == 0:
                partial_epoch = epoch + (batch_idx / len(self.train_loader))
                self.eval_epoch('val')
                print('______mid-epoch summary {0:.2f}/{1:.0f}______\n'.format(partial_epoch,self.nEpochs))
                self.writer.loss_summary()
                self.writer.accuracy_summary(mid_epoch=True)
                self.writer.save_history_to_csv()
                self.save_checkpoint_(partial_epoch)


    def eval(self,test=False):
        self.mode = 'test' if test else 'val'
        self.model = self.model.eval()

    def train(self):
        self.mode = 'train'
        self.model = self.model.train()

    def get_last_loss(self):
        if self.model.task == 'regression':
            return self.writer.val_MAE[-1]
        else:
            return self.writer.total_val_loss_history[-1]

    def get_last_accuracy(self):
        if hasattr(self.writer,'val_AUROC'):
            return self.writer.val_AUROC[-1]
        else:
            return None

    def save_checkpoint_(self,epoch):
        loss = self.get_last_loss()
        accuracy = self.get_last_accuracy()
        self.model.save_checkpoint(
            self.writer.experiment_folder, self.writer.experiment_title, epoch, loss ,accuracy, self.optimizer ,schedule=self.lr_handler.schedule)


    def forward_pass(self,input_dict):
        input_dict = {k:(v.cuda() if self.cuda else v) for k,v in input_dict.items()}
        output_dict = self.model(input_dict['fmri_sequence'])
        loss_dict, loss = self.aggregate_losses(input_dict, output_dict)
        if self.task == 'fine_tune':
            self.compute_accuracy(input_dict, output_dict)
        return loss_dict, loss


    def aggregate_losses(self,input_dict,output_dict):
        final_loss_dict = {}
        final_loss_value = 0
        for loss_name, current_loss_dict in self.writer.losses.items():
            if current_loss_dict['is_active']:
                loss_func = getattr(self, 'compute_' + loss_name)
                current_loss_value = loss_func(input_dict,output_dict)
                if current_loss_value.isnan().sum() > 0:
                    warnings.warn('found nans in computation')
                    print('at {} loss'.format(loss_name))
                lamda = current_loss_dict['factor']
                factored_loss = current_loss_value * lamda
                final_loss_dict[loss_name] = factored_loss.item()
                final_loss_value += factored_loss
        final_loss_dict['total'] = final_loss_value.item()
        return final_loss_dict, final_loss_value

    def compute_reconstruction(self,input_dict,output_dict):
        fmri_sequence = input_dict['fmri_sequence'][:,0].unsqueeze(1)
        reconstruction_loss = self.reconstruction_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return reconstruction_loss

    def compute_intensity(self,input_dict,output_dict):
        per_voxel = input_dict['fmri_sequence'][:,1,:,:,:,:]
        voxels = get_intense_voxels(per_voxel, output_dict['reconstructed_fmri_sequence'].shape)
        output_intense = output_dict['reconstructed_fmri_sequence'][voxels]
        truth_intense = input_dict['fmri_sequence'][:,0][voxels.squeeze(1)]
        intensity_loss = self.intensity_loss_func(output_intense.squeeze(), truth_intense)
        return intensity_loss

    def compute_perceptual(self,input_dict,output_dict):
        fmri_sequence = input_dict['fmri_sequence'][:,0].unsqueeze(1)
        perceptual_loss = self.perceptual_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return perceptual_loss

    def compute_binary_classification(self,input_dict,output_dict):
        binary_loss = self.binary_classification_loss_func(output_dict['binary_classification'].squeeze(), input_dict['subject_binary_classification'].squeeze())
        return binary_loss

    def compute_regression(self,input_dict,output_dict):
        gender_loss = self.regression_loss_func(output_dict['regression'].squeeze(),input_dict['subject_regression'].squeeze())
        return gender_loss

    def compute_accuracy(self,input_dict,output_dict):
        task = self.model.task
        out = output_dict[task].detach().clone().cpu()
        score = out.squeeze() if out.shape[0] > 1 else out
        labels = input_dict['subject_' + task].clone().cpu()
        subjects = input_dict['subject'].clone().cpu()
        for i, subj in enumerate(subjects):
            subject = str(subj.item())
            if subject not in self.writer.subject_accuracy:
                self.writer.subject_accuracy[subject] = {'score': score[i].unsqueeze(0), 'mode': self.mode, 'truth': labels[i],'count': 1}
            else:
                self.writer.subject_accuracy[subject]['score'] = torch.cat([self.writer.subject_accuracy[subject]['score'], score[i].unsqueeze(0)], dim=0)
                self.writer.subject_accuracy[subject]['count'] += 1

    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs


