import numpy as np
from torch.utils.data import DataLoader,Subset
from data_preprocess_and_load.datasets import *
from utils import reproducibility

class DataHandler():
    def __init__(self,test=False,**kwargs):
        self.test = test
        self.kwargs = kwargs
        self.dataset_name = kwargs.get('dataset_name')
        self.splits_folder = Path(kwargs.get('base_path')).joinpath('splits',self.dataset_name)
        self.splits_folder.mkdir(exist_ok=True)
        self.seed = kwargs.get('seed')
        self.current_split = self.splits_folder.joinpath('seed_{}.txt'.format(self.seed))

    def get_dataset(self):
        if self.dataset_name == 'S1200':
            return rest_1200_3D
        elif self.dataset_name == 'ucla':
            return ucla
        else:
            raise NotImplementedError

    def current_split_exists(self):
        return self.current_split.exists()

    def create_dataloaders(self):
        reproducibility(**self.kwargs)
        dataset = self.get_dataset()
        train_loader = dataset(**self.kwargs)
        eval_loader = dataset(**self.kwargs)
        eval_loader.augment = None
        self.subject_list = train_loader.index_l
        if self.current_split_exists():
            train_names, val_names, test_names = self.load_split()
            train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(train_names,val_names,test_names,self.subject_list)
        else:
            train_idx,val_idx,test_idx = self.determine_split_randomly(self.subject_list,**self.kwargs)

        # train_idx = [train_idx[x] for x in torch.randperm(len(train_idx))[:10]]
        # val_idx = [val_idx[x] for x in torch.randperm(len(val_idx))[:10]]

        train_loader = Subset(train_loader, train_idx)
        val_loader = Subset(eval_loader, val_idx)
        test_loader = Subset(eval_loader, test_idx)

        training_generator = DataLoader(train_loader, **self.get_params(**self.kwargs))
        val_generator = DataLoader(val_loader, **self.get_params(eval=True,**self.kwargs))
        test_generator = DataLoader(test_loader, **self.get_params(eval=True,**self.kwargs))  if self.test else None
        return training_generator, val_generator, test_generator


    def get_params(self,eval=False,**kwargs):
        batch_size = kwargs.get('batch_size')
        workers = kwargs.get('workers')
        cuda = kwargs.get('cuda')
        if eval:
            workers = 0
        params = {'batch_size': batch_size,
                  'shuffle': True,
                  'num_workers': workers,
                  'drop_last': True,
                  'pin_memory': False,  # True if cuda else False,
                  'persistent_workers': True if workers > 0 and cuda else False}
        return params

    def save_split(self,sets_dict):
        with open(self.current_split,'w+') as f:
            for name,subj_list in sets_dict.items():
                f.write(name + '\n')
                for subj_name in subj_list:
                    f.write(str(subj_name) + '\n')

    def convert_subject_list_to_idx_list(self,train_names,val_names,test_names,subj_list):
        subj_idx = np.array([str(x[0]) for x in subj_list])
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx,val_idx,test_idx

    def determine_split_randomly(self,index_l,**kwargs):
        train_percent = kwargs.get('train_split')
        val_percent = kwargs.get('val_split')
        S = len(np.unique([x[0] for x in index_l]))
        S_train = int(S * train_percent)
        S_val = int(S * val_percent)
        S_train = np.random.choice(S, S_train, replace=False)
        remaining = np.setdiff1d(np.arange(S), S_train)
        S_val = np.random.choice(remaining,S_val, replace=False)
        S_test = np.setdiff1d(np.arange(S), np.concatenate([S_train,S_val]))
        train_idx,val_idx,test_idx = self.convert_subject_list_to_idx_list(S_train,S_val,S_test,self.subject_list)
        self.save_split({'train_subjects':S_train,'val_subjects':S_val,'test_subjects':S_test})
        return train_idx,val_idx,test_idx

    def load_split(self):
        subject_order = open(self.current_split, 'r').readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(['train' in line for line in subject_order])
        val_index = np.argmax(['val' in line for line in subject_order])
        test_index = np.argmax(['test' in line for line in subject_order])
        train_names = subject_order[train_index + 1:val_index]
        val_names = subject_order[val_index+1:test_index]
        test_names = subject_order[test_index + 1:]
        return train_names,val_names,test_names