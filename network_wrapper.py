import bdlb
import matplotlib.pyplot as plt
import numpy as np

import torch
import random
import argparse
import torch
import sys



random_seed = 7777
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

import leek_anomaly.network as network
from leek_anomaly.config import cfg, assert_and_infer_cfg
import leek_anomaly.optimizer as optimizer


def visualize_tfdataset(tfdataset, num_samples):
    """Visualizes `num_samples` from the `tfdataset`."""
    fig, axs = plt.subplots(num_samples, 2, figsize=(7, 2*num_samples))
    for i, blob in enumerate(tfdataset.take(num_samples)):
        image = blob['image_left'].numpy()
    
        mask = blob['mask'].numpy()
        axs[i][0].imshow(image.astype('int'))
        axs[i][0].axis("off")
        axs[i][0].set(title="Image")

        mask[mask == 255] = 2
        mask[mask == 1] = 1
    
        axs[i][1].imshow(mask[..., 0])
        axs[i][1].axis("off")
        axs[i][1].set(title="Mask")
    fig.show()


parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR101V3PlusD_OS8',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', nargs='*', type=str, default='cityscapes',
                    help='a list of datasets; cityscapes')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default='cityscapes',
                    help='a list consists of cityscapes')
parser.add_argument('--val_interval', type=int, default=100000, help='validation interval')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)
parser.add_argument('--freeze_trunk', action='store_true', default=False)

parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default='data/r101_os8_base_cty.pth')
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--backbone_lr', type=float, default=0.0,
                    help='different learning rate on backbone network')

parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling methods, average is better than max')

parser.add_argument('--ood_dataset_path', type=str,
                    default='/home/nas1_userB/dataset/ood_segmentation/fishyscapes',
                    help='OoD dataset path')

# Anomaly score mode - msp, max_logit, standardized_max_logit
parser.add_argument('--score_mode', type=str, default='standardized_max_logit',
                    help='score mode for anomaly [msp, max_logit, standardized_max_logit]')

# Boundary suppression configs
parser.add_argument('--enable_boundary_suppression', type=bool, default=True,
                    help='enable boundary suppression')
parser.add_argument('--boundary_width', type=int, default=4,
                    help='initial boundary suppression width')
parser.add_argument('--boundary_iteration', type=int, default=4,
                    help='the number of boundary iterations')

# Dilated smoothing configs
parser.add_argument('--enable_dilated_smoothing', type=bool, default=True,
                    help='enable dilated smoothing')
parser.add_argument('--smoothing_kernel_size', type=int, default=7,
                    help='kernel size of dilated smoothing')
parser.add_argument('--smoothing_kernel_dilation', type=int, default=6,
                    help='kernel dilation rate of dilated smoothing')

#Trying to disable argparse
args = parser.parse_args(['--lr','0.01'])


# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True


import time

args.world_size = 1

print(f'World Size: {args.world_size}')
torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)
torch.distributed.init_process_group(backend='nccl',
                                     init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.local_rank)


def get_net(args):
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)

    net = network.get_net(args, criterion=None, criterion_aux=None)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    
    import leek_anomaly
    import os
    
    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, None, None,
                            os.path.dirname(os.path.abspath(leek_anomaly.__file__))+'/'+args.snapshot, args.restore_optimizer)
        print(f"Loading completed. Epoch {epoch} and mIoU {mean_iu}")
    else:
        raise ValueError(f"snapshot argument is not set!")

    class_mean = np.load(os.path.dirname(os.path.abspath(leek_anomaly.__file__))+'/'+f'data/cityscapes_mean_reported.npy', allow_pickle=True)
    class_var = np.load(os.path.dirname(os.path.abspath(leek_anomaly.__file__))+'/'+f'data/cityscapes_var_reported.npy', allow_pickle=True)
    net.module.set_statistics(mean=class_mean.item(), var=class_var.item())

    torch.cuda.empty_cache()
    net.eval()

    return net

net = get_net(args)


features_t = []
def hook_t(module, input, output):
    features_t.append(output.cuda())

net.module.final1.register_forward_hook(hook_t)
net.module.final2.register_forward_hook(hook_t)

import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
import shutil
import time
from torchvision.models import resnet18, resnet101
from PIL import Image
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def val_data_transforms(mean_train=mean_train, std_train=std_train):
    data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train,
                                std=std_train)
            ])
    return data_transforms


def estimator(image):
    global num
    global features_t

    features_t = []
    """Assigns a random uncertainty per pixel."""
    val_data_transform = val_data_transforms(mean_train=mean_train, std_train=std_train)
    test_img = Image.fromarray(np.array(image))
    test_img = val_data_transform(test_img)
    test_img = torch.unsqueeze(test_img, 0).to(device)

    with torch.set_grad_enabled(False):
        out, out2 = net(test_img)
   
    logit2, predction01 = features_t[1].detach().cpu().max(1)
    new_logit = -out2.clone()
    
    new_logit = new_logit.unsqueeze(1)
    new_logit = F.interpolate(new_logit, size=(256, 512), mode='bilinear', align_corners=True)
    new_logit = new_logit.squeeze(1)
    feats =  features_t[0]
    new_logit -= torch.min(new_logit)
    new_logit /= torch.max(new_logit)

    maps = torch.ones(1, 256, 512)
    
    for i in range(3):
        feats = (((1-new_logit.cuda()) * feats) + ((new_logit.cuda()) * torch.max(feats) ))
        output = net.module.final2(feats)
        anomaly_score, _ = output.detach().max(1)
        new_logit = anomaly_score.detach().cpu()
        
        new_logit = new_logit.detach().cpu()[0]
        new_logit -= torch.min(new_logit)
        new_logit /= torch.max(new_logit)
        
        maps[:] = new_logit
        
    new_logit = maps
    new_logit = new_logit.unsqueeze(1)
    new_logit = F.interpolate(new_logit, size=(1024, 2048), mode='bilinear', align_corners=True)
    new_logit = new_logit.squeeze(1)
    
    out3 = -out2.detach().cpu()[0]

    new_logit = new_logit[0]
    new_logit -= torch.min(new_logit)
    new_logit /= torch.max(new_logit)
    
    norm_logit = new_logit

    final_output = out3 * (1-norm_logit)
    
    del out
    return torch.tensor(final_output.detach().cpu())


if __name__ == '__main__':
    fs = bdlb.load(benchmark="fishyscapes", download_and_prepare=False)
    ds = fs.get_dataset('LostAndFound')
    metrics = fs.evaluate(estimator, ds.take(100))
    print("AP: " + str(metrics['AP']) + " FPR95: " + str(metrics['FPR@95%TPR']) +  " auroc: " + str(metrics['auroc']))

    