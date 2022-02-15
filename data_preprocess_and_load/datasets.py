import os
import torch
from torch.utils.data import Dataset
import augmentations
import pandas as pd
from pathlib import Path


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
    def register_args(self,**kwargs):
        #todo:decide if keep immedieate load or not
        self.device = None#torch.device('cuda') if kwargs.get('cuda') else torch.device('cpu')
        self.index_l = []
        self.norm = 'global_normalize'
        self.complementary = 'per_voxel_normalize'
        self.random_TR = kwargs.get('random_TR')
        self.set_augmentations(**kwargs)
        self.stride_factor = 1
        self.sequence_stride = 1
        self.sequence_length = kwargs.get('sequence_length')
        self.sample_duration = self.sequence_length * self.sequence_stride
        self.stride = max(round(self.stride_factor * self.sample_duration),1)
        self.TR_skips = range(0,self.sample_duration,self.sequence_stride)

    def get_input_shape(self):
        shape = torch.load(os.path.join(self.index_l[0][2],self.index_l[0][3] + '.pt')).squeeze().shape
        return shape

    def set_augmentations(self,**kwargs):
        if kwargs.get('augment_prob') > 0:
            self.augment = augmentations.brain_gaussian(**kwargs)
        else:
            self.augment = None

    def TR_string(self,filename_TR,x):
        #all datasets should have the TR mentioned in the format of 'some prefix _ number.pt'
        TR_num = [xx for xx in filename_TR.split('_') if xx.isdigit()][0]
        assert len(filename_TR.split('_')) == 2
        filename = filename_TR.replace(TR_num,str(int(TR_num) + x)) + '.pt'
        return filename

    def determine_TR(self,TRs_path,TR):
        if self.random_TR:
            possible_TRs = len(os.listdir(TRs_path)) - self.sample_duration
            TR = 'TR_' + str(torch.randint(0,possible_TRs,(1,)).item())
        return TR

    def load_sequence(self, TRs_path, TR):
        # the logic of this function is that always the first channel corresponds to global norm and if there is a second channel it belongs to per voxel.
        TR = self.determine_TR(TRs_path,TR)
        y = torch.cat([torch.load(os.path.join(TRs_path, self.TR_string(TR, x)),map_location=self.device).unsqueeze(0) for x in self.TR_skips], dim=4)
        if self.complementary is not None:
            y1 = torch.cat([torch.load(os.path.join(TRs_path, self.TR_string(TR, x)).replace(self.norm, self.complementary),map_location=self.device).unsqueeze(0)
                            for x in self.TR_skips], dim=4)
            y1[y1!=y1] = 0
            y = torch.cat([y, y1], dim=0)
            del y1
        return y

class rest_1200_3D(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.root = r'D:\users\Gony\HCP-1200'
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','HCP_1200_gender.csv'))
        self.meta_data_residual = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','HCP_1200_precise_age.csv'))
        self.data_dir = os.path.join(self.root, 'MNI_to_TRs')
        self.subject_names = os.listdir(self.data_dir)
        self.label_dict = {'F': torch.tensor([0.0]), 'M': torch.tensor([1.0]), '22-25': torch.tensor([1.0, 0.0]),
                           '26-30': torch.tensor([1.0, 0.0]),
                           '31-35': torch.tensor([0.0, 1.0]), '36+': torch.tensor([0.0, 1.0])}  # torch.tensor([1])}
        self.subject_folders = []
        for i,subject in enumerate(os.listdir(self.data_dir)):
            try:
                age = torch.tensor(self.meta_data_residual[self.meta_data_residual['subject']==int(subject)]['age'].values[0])
            except Exception:
                #deal with discrepency that a few subjects don't have exact age, so we take the mean of the age range as the exact age proxy
                age = self.meta_data[self.meta_data['Subject'] == int(subject)]['Age'].values[0]
                age = torch.tensor([float(x) for x in age.replace('+','-').split('-')]).mean()
            gender = self.meta_data[self.meta_data['Subject']==int(subject)]['Gender'].values[0]
            path_to_TRs = os.path.join(self.data_dir,subject,self.norm)
            subject_duration = len(os.listdir(path_to_TRs))#121
            session_duration = subject_duration - self.sample_duration
            filename = os.listdir(path_to_TRs)[0]
            filename = filename[:filename.find('TR')+3]

            for k in range(0,session_duration,self.stride):
                self.index_l.append((i, subject, path_to_TRs,filename + str(k),session_duration, age , gender))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_TRs, TR , session_duration, age, gender = self.index_l[index]
        age = self.label_dict[age] if isinstance(age,str) else age.float()
        y = self.load_sequence(path_to_TRs,TR)
        if self.augment is not None:
            y = self.augment(y)
        return {'fmri_sequence':y,'subject':subj,'subject_binary_classification':self.label_dict[gender],'subject_regression':age,'TR':int(TR.split('_')[1])}


class ucla(BaseDataset):
    def __init__(self, **kwargs):
        super(ucla, self).__init__()
        self.register_args(**kwargs)
        datasets_folder = str(Path(kwargs.get('base_path')).parent.parent)
        self.root = os.path.join(datasets_folder,'fmri_data','ucla','ucla','output')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ucla_participants.tsv'),sep='\t')
        self.data_dir = os.path.join(self.root, 'rest')
        self.subjects = len(os.listdir(self.data_dir))
        self.subjects_names = os.listdir(self.data_dir)
        for i, subject in enumerate(self.subjects_names):
            try:
                diagnosis = self.meta_data.loc[self.meta_data['participant_id'] == subject, ['diagnosis']].values[0][0]
            except Exception as e:
                print(e)
            TRs_path = os.path.join(self.data_dir, subject,self.norm)
            session_duration = len(os.listdir(TRs_path)) - self.sample_duration
            diagnosis = torch.tensor([0.0]) if diagnosis == 'CONTROL' else torch.tensor([1.0])
            #TODO:debug
            #diagnosis = torch.tensor([1.0, 0.0]) if diagnosis == 'CONTROL' else torch.tensor([0.0, 1.0])
            for k in range(0, session_duration, self.stride):
                self.index_l.append((i, subject, TRs_path, 'TR_' + str(k), session_duration, diagnosis ))


    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj_num, subj_name ,TRs_path, TR, session_duration, diagnosis = self.index_l[index]
        y = self.load_sequence(TRs_path,TR)
        if self.augment is not None:
            y = self.augment(y)
        input_dict = {'fmri_sequence':y,'subject':subj_num ,'subject_binary_classification':diagnosis , 'TR':int(TR.split('_')[1])}
        return input_dict

class ptsd(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.sessions = ['ses-1','ses-2','ses-3']
        self.root = r'D:\users\Gony\ptsd\ziv'
        self.meta_data = pd.read_csv(os.path.join(self.root, 'caps.csv'))
        self.data_dir = os.path.join(self.root, 'MNI_to_TRs')
        self.subject_names = os.listdir(self.data_dir)
        self.subjects = len(os.listdir(self.data_dir))
        self.index_l = []
        for i,subject in enumerate(os.listdir(self.data_dir)):
            for session in os.listdir(os.path.join(self.data_dir,subject)):
                for task in os.listdir(os.path.join(self.data_dir, subject,session)):
                    ses = str(session[session.find('-')+1])
                    category1 = "T" + ses + "_TotalCaps4'"
                    category2 = "T" + ses + "_TotalCaps5'"
                    category3 = "T" + ses + "_Is PTSD_Final"
                    score = self.meta_data.loc[self.meta_data['Subject ID'] == int(subject[-4:]),[category1,category2,category3]].values.tolist()[0]
                    TRs_path = os.path.join(self.data_dir, subject,session,task, self.norm_name)
                    session_duration = len(os.listdir(TRs_path)) - self.sample_duration
                    for k in range(0,session_duration, self.stride):
                        if not any(np.isnan(score)):
                            self.index_l.append((i, subject[-4:], TRs_path, 'TR_' + str(k), session_duration, (task, session, score[0], score[1], score[2])))
        if not self.fine_tune:
            extra_data = self.data_dir.replace('ziv','tom')
            for j,subj in enumerate(os.listdir(extra_data)):
                for time in os.listdir(os.path.join(extra_data,subj)):
                    for session in os.listdir(os.path.join(extra_data,subj,time)):
                        for task in os.listdir(os.path.join(extra_data,subj,time,session)):
                            path_to_TRs = os.path.join(extra_data,subj,time,session,task,self.norm_name)
                            session_duration = len(os.listdir(path_to_TRs)) - self.sample_duration
                            for k in range(0,session_duration,self.stride):
                                self.index_l.append((j+i,subj,path_to_TRs, 'TR_' + str(k), session_duration, (task,session,np.nan,np.nan,np.nan)))
            extra_data = self.data_dir.replace(r'ptsd\ziv','ayam')
            for k, subject in enumerate(os.listdir(extra_data)):
                for task in os.listdir(os.path.join(extra_data, subject)):
                    TRs_path = os.path.join(extra_data, subject, task, self.norm_name)
                    session_duration = len(os.listdir(TRs_path)) - self.sample_duration
                    for kk in range(0, session_duration, self.stride):
                        self.index_l.append((k, subject, TRs_path, 'TR_' + str(kk), session_duration, (task,np.nan, np.nan, np.nan, np.nan)))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj_num, subj_name ,TRs_path, TR, session_duration, diagnosis = self.index_l[index]
        y = self.load_sequence(TRs_path,TR)
        if self.augment is not None:
            y = self.augment(y)
        input_dict = {'fmri_sequence':y,'subject':subj_num ,'subject_binary_classification':diagnosis , 'TR':int(TR.split('_')[1])}
        return input_dict