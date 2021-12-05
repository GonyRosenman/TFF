from utils import *
from loss_writer import Writer
from losses import Percept_Loss
import torch
from torch.nn import MSELoss,L1Loss,BCELoss,CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
import argparse
seed = 555555555

def get_arguments(base_path,seed):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=seed)
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dim', default=(75,91,83))
    parser.add_argument('--validation_frequency', type=int, default=3000)
    parser.add_argument('--dataset_name', type=str, default="ucla")
    parser.add_argument('--nEpochs', type=int, default=20)
    parser.add_argument('--augment', type=str, default={'gaussian':{'blur':{'prob':0,'sigma':(0.0,1)},'noise':{'prob':0,'scale':(0,0.3)}}})
    parser.add_argument('--task', type=str, default='fine_tune')
    parser.add_argument('--running_mean_size', default=1000)
    parser.add_argument('--random_TR', default=True)
    parser.add_argument('--lr', default={'base_lr':3e-5,'step_size':1000,'LRgamma':0.9,'final_lr':1e-5})
    parser.add_argument('--losses', default={'intensity':{'is_active':False,'criterion':L1Loss(),'thresholds':[0.9,0.99,15],'type':'absolute','factor':1,'erosion':False},
                                             'perceptual':{'is_active':False,'loss_type':'MSE','memory_percent':0.5,'method':'repeat','factor':1,'layers':[True,True,False,False]},
                                             'reconstruction':{'is_active':False,'criterion':L1Loss(),'factor':1},
                                             'gender':{'is_active':False,'criterion':BCELoss(),'factor':1},
                                             'age':{'is_active':False,'criterion':L1Loss(),'factor':1},
                                             'diagnosis':{'is_active':True,'criterion':BCELoss(),'factor':1}})
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--workers', default=2)
    parser.add_argument('--finished', default=False)
    parser.add_argument('--stride', default={'sample_stride_factor':1,'sequence_stride':1,'sequence_length':20,'random_TR':True})
    parser.add_argument('--nvidia_vars', default={'is_nvidia':True,'model_depth':4,'hidden_layers':2,'intermediate_vec':2640,
                                                  'extract_vec_method':'cls','dropout':{'input': 0.2, 'green': 0.4,'Up_green': 0.2}})
    parser.add_argument('--fine_tune', default=True)
    parser.add_argument('--fine_tune_task', default='diagnosis')
    parser.add_argument('--weight_decay', default=1e-2)
    parser.add_argument('--log_dir', type=str,default=os.path.join(base_path,'runs'))

    args = parser.parse_args()
    args.ID = datestamp()
    args.title = 'test____{}_{}_{}'.format(args.dataset_name,args.task,args.ID)
    args.save = os.path.join(base_path,'experiments', args.title)

    return args


def run(old_args, base_path, cuda_num):
    cuda_num = str(cuda_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num
    args = get_arguments(base_path,seed)
    _, _, test_loader = create_dataloaders(args,test=True)
    args.dim = test_loader.dataset.dataset.get_input_shape()
    metric = 'loss' if old_args.fine_tune_task == 'age' else 'accuracy'
    model_weights = os.path.join(old_args.save,old_args.title + '_BEST_val_{}.pth'.format(metric))
    model,_ = create_model_and_optimizer(args, model_weights, args.lr['base_lr'])
    writer = Writer(args, model, sets=['test'])
    writer.eval(test=True)

    for batch_idx, input_dict in enumerate(tqdm(test_loader,position=0,leave=True)):
        with torch.no_grad():
            loss_dict,loss = writer.forward_pass(input_dict)
        writer.write_losses(loss_dict,set='test')
    writer.loss_summary()
    writer.accuracy_summary(mid_epoch=False)
    print('final test results:')
    for attribute in dir(writer):
        if 'test' in attribute and not 'loss' in attribute:
            metric_value = getattr(writer,attribute)[-1]
            print('{} - {}'.format(attribute,metric_value))
    writer.save_history_to_csv()
