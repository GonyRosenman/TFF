import numpy as np
from torch.utils.data import DataLoader,Subset
from data_preprocess_and_load.datasets import *
from utils import reproducibility

def get_params(**kwargs):
    batch_size = kwargs.get('batch_size')
    workers = kwargs.get('workers')
    cuda = kwargs.get('cuda')
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': workers,
              'drop_last': True,
              'pin_memory': False,#True if cuda else False,
              'persistent_workers': True if workers > 0 and cuda else False}
    return params


def determine_split_randomly(index_l,**kwargs):
    split_percent = kwargs.get('pretrain_split')
    S = len(np.unique([x[0] for x in index_l]))
    S_train = int(S * split_percent)
    S_train = np.random.choice(S, S_train, replace=False)
    S_val = np.setdiff1d(np.arange(S), S_train)
    subj_idx = np.array([x[0] for x in index_l])
    train_idx = np.where(np.in1d(subj_idx, S_train))[0].tolist()
    val_idx = np.where(np.in1d(subj_idx, S_val))[0].tolist()
    return train_idx,val_idx,None

def predetermined_split(subject_order,index_l):
    subject_order = [x[:-1] for x in subject_order]
    train_names = subject_order[np.argmax(['train' in line for line in subject_order]) + 1:np.argmax(
        ['test' in line for line in subject_order])]
    val_names = subject_order[np.argmax(['val' in line for line in subject_order]) + 1:]
    test_names = subject_order[np.argmax(['test' in line for line in subject_order]) + 1:np.argmax(
        ['val' in line for line in subject_order])]
    subj_names = np.array([x[1] for x in index_l])
    train_idx = np.where(np.in1d(subj_names, train_names))[0].tolist()
    train_idx = np.random.permutation(train_idx)
    val_idx = np.where(np.in1d(subj_names, val_names))[0].tolist()
    val_idx = np.random.permutation(val_idx)
    test_idx = np.where(np.in1d(subj_names, test_names))[0].tolist()
    test_idx = np.random.permutation(test_idx)
    return train_idx,val_idx,test_idx

def create_dataloaders(sets,**kwargs):
    test = any([set == 'test' for set in sets])
    reproducibility(**kwargs)
    params = get_params(**kwargs)
    dataset_name = kwargs.get('dataset_name')
    datasets_dict = {'S1200':{'final_split_path':r'D:\users\Gony\HCP-1200\final_split_train_test.txt','loader':rest_1200_3D},
                     'ucla':{'final_split_path':r'D:\users\Gony\ucla\ucla\ucla\output\final_split_train_test.txt','loader':ucla}}

    train_loader = datasets_dict[dataset_name]['loader'](**kwargs)
    eval_loader = datasets_dict[dataset_name]['loader'](**kwargs)
    eval_loader.augment = None
    if not kwargs.get('task') == 'fine_tune':
        train_idx,val_idx,test_idx = determine_split_randomly(train_loader.index_l,**kwargs)
    else:
        subject_order = open(datasets_dict[dataset_name]['final_split_path'], 'r').readlines()
        train_idx, val_idx, test_idx = predetermined_split(subject_order,train_loader.index_l)

    train_loader = Subset(train_loader,train_idx)
    val_loader = Subset(eval_loader,val_idx)
    test_loader = Subset(eval_loader, test_idx) if test else None

    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)
    test_generator = DataLoader(test_loader, **params) if test else None
    return training_generator, val_generator , test_generator
