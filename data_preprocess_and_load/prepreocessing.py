import os
import numpy as np
import nibabel as nib
import torch
from multiprocessing import Process, Queue


def read_hcp(file_path,global_norm_path,per_voxel_norm_path,hand,count,queue=None):
    img_orig = torch.from_numpy(np.asanyarray(nib.load(file_path).dataobj)[8:-8, 8:-8, :-10, 10:]).to(dtype=torch.float32)
    background = img_orig == 0
    img_temp = (img_orig - img_orig[~background].mean()) / (img_orig[~background].std())
    img = torch.empty(img_orig.shape)
    img[background] = img_temp.min()
    img[~background] = img_temp[~background]
    img = torch.split(img, 1, 3)
    for i, TR in enumerate(img):
        torch.save(TR.clone(),
                   os.path.join(global_norm_path, 'rfMRI_' + hand + '_TR_' + str(i) + '.pt'))
    # repeat for per voxel normalization
    img_temp = (img_orig - img_orig.mean(dim=3, keepdims=True)) / (img_orig.std(dim=3, keepdims=True))
    img = torch.empty(img_orig.shape)
    img[background] = img_temp.min()
    img[~background] = img_temp[~background]
    img = torch.split(img, 1, 3)
    for i, TR in enumerate(img):
        torch.save(TR.clone(),
                   os.path.join(per_voxel_norm_path, 'rfMRI_' + hand + '_TR_' + str(i) + '.pt'))
    print('finished another subject. count is now {}'.format(count))

def main():
    hcp_path = r'D:\users\Gony\HCP-1200'
    all_files_path = os.path.join(hcp_path,'extract_S1200_data')
    queue = Queue()
    count = 0
    for subj in os.listdir(all_files_path):
        subj_path = os.path.join(all_files_path,subj)
        try:
            file_path = os.path.join(subj_path,os.listdir(subj_path)[0])
            hand = file_path[file_path.find('REST1_')+6:file_path.find('.nii')]
            global_norm_path = os.path.join(hcp_path,'MNI_to_TRs',subj,'global_normalize')
            per_vox_norm_path = os.path.join(hcp_path, 'MNI_to_TRs', subj, 'per_voxel_normalize')
            os.makedirs(global_norm_path, exist_ok=True)
            os.makedirs(per_vox_norm_path, exist_ok=True)
            count += 1
            print('start working on subject '+ subj)
            p = Process(target=read_hcp, args=(file_path,global_norm_path,per_vox_norm_path,hand,count, queue))
            p.start()
            if count % 20 == 0:
                p.join()  # this blocks until the process terminates
        except Exception:
            print('encountered problem with '+subj)
            print(Exception)
if __name__ == '__main__':
    main()