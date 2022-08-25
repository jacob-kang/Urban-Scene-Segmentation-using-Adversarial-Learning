"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import sys
import time
import torch

#from torch.cuda import amp
from apex import amp
from runx.logx import logx
from config import assert_and_infer_cfg, update_epoch, cfg
from utils.misc import AverageMeter, prep_experiment, eval_metrics, neptune_logger
from utils.misc import ImageDumper
from utils.trnval_utils import eval_minibatch, validate_topn
from loss.utils import get_loss
from loss.optimizer import get_optimizer, restore_opt, restore_net,forgiving_state_restore

import datasets
import network

import torch.nn as nn
import numpy as np

import torch.nn.functional

# Import autoresume module
sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
AutoResume = None
try:
    from userlib.auto_resume import AutoResume
except ImportError:
    print(AutoResume)


parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--arch', type=str, default='deepv3.DeepV3PlusR50',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, mapillary, camvid, kitti')
parser.add_argument('--dataset_inst', default=None,
                    help='placeholder for dataset instance')
parser.add_argument('--num_workers', type=int, default=4,
                    help='cpu worker threads per dataloader instance')

parser.add_argument('--cv', type=int, default=0,
                    help=('Cross-validation split id to use. Default # of splits set'
                          ' to 3 in config'))

parser.add_argument('--class_uniform_pct', type=float, default=0.5,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='Use coarse annotations for specific classes')

parser.add_argument('--custom_coarse_dropout_classes', type=str, default=None,
                    help='Drop some classes from auto-labelling')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--rmi_loss', action='store_true', default=False,
                    help='use RMI loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help=('Batch weighting for class (use nll class weighting using '
                          'batch stats'))

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_epoch', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new lr ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--apex', action='store_true', default=False,
                    help='Use Nvidia Apex Distributed Data Parallel')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')

parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')
parser.add_argument('--global_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--amsgrad', action='store_true', help='amsgrad for adam')

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help=('0 means no aug, 1 means hard negative mining '
                          'iter 1, 2 means hard negative mining iter 2'))

parser.add_argument('--trunk', type=str, default='resnet50',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_cu_epoch', type=int, default=150,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--brt_aug', action='store_true', default=False,
                    help='Use brightness augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=2.0,
                    help='polynomial LR exponent')
parser.add_argument('--poly_step', type=int, default=110,
                    help='polynomial epoch step')
parser.add_argument('--bs_trn', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=str, default="768",
                    help=('training crop size: either scalar or h,w'))
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--resume', type=str, default=None,
                    help=('continue training from a checkpoint. weights, '
                          'optimizer, schedule are restored'))
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--restore_net', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--result_dir', type=str, default='./logs',
                    help='where to write log output')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help=('Minimum testing to verify nothing failed, '
                          'Runs code for 1 epoch of train and val'))
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
# Full Crop Training
parser.add_argument('--full_crop_training', action='store_true', default=False,
                    help='Full Crop Training')

# Multi Scale Inference
parser.add_argument('--multi_scale_inference', action='store_true',
                    help='Run multi scale inference')

parser.add_argument('--default_scale', type=float, default=1.0,
                    help='default scale to run validation')

parser.add_argument('--log_msinf_to_tb', action='store_true', default=False,
                    help='Log multi-scale Inference to Tensorboard')

parser.add_argument('--eval', type=str, default=None,
                    help=('just run evaluation, can be set to val or trn or '
                          'folder'))
parser.add_argument('--eval_folder', type=str, default=None,
                    help='path to frames to evaluate')
parser.add_argument('--three_scale', action='store_true', default=False)
parser.add_argument('--alt_two_scale', action='store_true', default=False)
parser.add_argument('--do_flip', action='store_true', default=False)
parser.add_argument('--extra_scales', type=str, default='0.5,2.0')
parser.add_argument('--n_scales', type=str, default=None)
parser.add_argument('--align_corners', action='store_true',
                    default=False)
parser.add_argument('--translate_aug_fix', action='store_true', default=False)
parser.add_argument('--mscale_lo_scale', type=float, default=0.5,
                    help='low resolution training scale')
parser.add_argument('--pre_size', type=int, default=None,
                    help=('resize long edge of images to this before'
                          ' augmentation'))
parser.add_argument('--amp_opt_level', default='O1', type=str,
                    help=('amp optimization level'))
parser.add_argument('--rand_augment', default=None,
                    help='RandAugment setting: set to \'N,M\'')
parser.add_argument('--init_decoder', default=False, action='store_true',
                    help='initialize decoder with kaiming normal')
parser.add_argument('--dump_topn', type=int, default=0,
                    help='Dump worst val images')
parser.add_argument('--dump_assets', action='store_true',
                    help='Dump interesting assets')
parser.add_argument('--dump_all_images', action='store_true',
                    help='Dump all images, not just a subset')
parser.add_argument('--dump_for_submission', action='store_true',
                    help='Dump assets for submission')
parser.add_argument('--dump_for_auto_labelling', action='store_true',
                    help='Dump assets for autolabelling')
parser.add_argument('--dump_topn_all', action='store_true', default=False,
                    help='dump topN worst failures')
parser.add_argument('--custom_coarse_prob', type=float, default=None,
                    help='Custom Coarse Prob')
parser.add_argument('--only_coarse', action='store_true', default=False)
parser.add_argument('--mask_out_cityscapes', action='store_true',
                    default=False)
parser.add_argument('--ocr_aspp', action='store_true', default=False)
parser.add_argument('--map_crop_val', action='store_true', default=False)
parser.add_argument('--aspp_bot_ch', type=int, default=None)
parser.add_argument('--trial', type=int, default=None)
parser.add_argument('--mscale_cat_scale_flt', action='store_true',
                    default=False)
parser.add_argument('--mscale_dropout', action='store_true',
                    default=False)
parser.add_argument('--mscale_no3x3', action='store_true',
                    default=False, help='no inner 3x3')
parser.add_argument('--mscale_old_arch', action='store_true',
                    default=False, help='use old attention head')
parser.add_argument('--mscale_init', type=float, default=None,
                    help='default attention initialization')
parser.add_argument('--attnscale_bn_head', action='store_true',
                    default=False)
parser.add_argument('--set_cityscapes_root', type=str, default=None,
                    help='override cityscapes default root dir')
parser.add_argument('--ocr_alpha', type=float, default=None,
                    help='set HRNet OCR auxiliary loss weight')
parser.add_argument('--val_freq', type=int, default=1,
                    help='how often (in epochs) to run validation')
parser.add_argument('--deterministic', action='store_true',
                    default=False)
parser.add_argument('--summary', action='store_true',
                    default=False)
parser.add_argument('--segattn_bot_ch', type=int, default=None,
                    help='bottleneck channels for seg and attn heads')
parser.add_argument('--grad_ckpt', action='store_true',
                    default=False)
parser.add_argument('--no_metrics', action='store_true', default=False,
                    help='prevent calculation of metrics')
parser.add_argument('--supervised_mscale_loss_wt', type=float, default=None,
                    help='weighting for the supervised loss')
parser.add_argument('--ocr_aux_loss_rmi', action='store_true', default=False,
                    help='allow rmi for aux loss')


args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}


#jacob import
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from neptune_token import run       #netpune token.

# Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ and args.apex:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.global_rank = int(os.environ['RANK'])

if args.apex:
    print('Global Rank: {} Local Rank: {}'.format(
        args.global_rank, args.local_rank))
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

#--------------------------
#GAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.cnn_part = nn.Sequential(
            nn.Conv2d(in_channels=19,out_channels=30,kernel_size=7,stride=2),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.Conv2d(in_channels=30,out_channels=60,kernel_size=3,stride=2),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.Conv2d(in_channels=60,out_channels=120,kernel_size=3,stride=2),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fclayer_part = nn.Sequential(
            nn.Linear(120, 300),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(300, 150),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(150, 1),
            #nn.Sigmoid(),
        )

    def forward(self, img):
        cnns =self.cnn_part(img)
        flatten_cnns = torch.flatten(cnns,1)
        validity = self.fclayer_part(flatten_cnns)

        return validity
#----------------------------




def check_termination(epoch):
    if AutoResume:
        shouldterminate = AutoResume.termination_requested()
        if shouldterminate:
            if args.global_rank == 0:
                progress = "Progress %d%% (epoch %d of %d)" % (
                    (epoch * 100 / args.max_epoch),
                    epoch,
                    args.max_epoch
                )
                AutoResume.request_resume(
                    user_dict={"RESUME_FILE": logx.save_ckpt_fn,
                               "TENSORBOARD_DIR": args.result_dir,
                               "EPOCH": str(epoch)
                               }, message=progress)
                return 1
            else:
                return 1
    return 0


def main():
    """
    Main Function
    """
    if AutoResume:
        AutoResume.init()

    assert args.result_dir is not None, 'need to define result_dir arg'
    logx.initialize(logdir=args.result_dir,
                    tensorboard=True, hparams=vars(args),
                    global_rank=args.global_rank)

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    prep_experiment(args)
    train_loader, val_loader, train_obj = \
        datasets.setup_loaders(args)
    criterion, criterion_val = get_loss(args)   #CrossEntropyLoss2d

    auto_resume_details = None
    if AutoResume:
        auto_resume_details = AutoResume.get_resume_details()

    if auto_resume_details:
        checkpoint_fn = auto_resume_details.get("RESUME_FILE", None)
        checkpoint = torch.load(checkpoint_fn,
                                map_location=torch.device('cpu'))
        args.result_dir = auto_resume_details.get("TENSORBOARD_DIR", None)
        args.start_epoch = int(auto_resume_details.get("EPOCH", None)) + 1
        args.restore_net = True
        args.restore_optimizer = True
        msg = ("Found details of a requested auto-resume: checkpoint={}"
               " tensorboard={} at epoch {}")
        logx.msg(msg.format(checkpoint_fn, args.result_dir,
                            args.start_epoch))
    elif args.resume:
        checkpoint = torch.load(args.resume,
                                map_location=torch.device('cpu'))
        args.arch = checkpoint['arch']
        args.start_epoch = int(checkpoint['epoch']) + 1
        args.restore_net = True
        args.restore_optimizer = True
        msg = "Resuming from: checkpoint={}, epoch {}, arch {}"
        logx.msg(msg.format(args.resume, args.start_epoch, args.arch))
    elif args.snapshot:
        if 'ASSETS_PATH' in args.snapshot:
            args.snapshot = args.snapshot.replace('ASSETS_PATH', cfg.ASSETS_PATH)
        checkpoint = torch.load(args.snapshot,
                                map_location=torch.device('cpu'))
        args.restore_net = True
        msg = "Loading weights from: checkpoint={}".format(args.snapshot)
        logx.msg(msg)

    net = network.get_net(args, criterion)      #model retrun (deeplabv3 resnet-50)
    optim, scheduler = get_optimizer(args, net) #optim : SGD

    discriminator = Discriminator().cuda()     #GAN
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.00001)       #GAN

    if args.restore_optimizer:
        restore_opt(optim, checkpoint)
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])

    if args.restore_net:
        restore_net(net, checkpoint)
        forgiving_state_restore(discriminator, checkpoint['discriminator'])


    if args.init_decoder:
        net.module.init_mods()

    torch.cuda.empty_cache()

    if args.start_epoch != 0:
        scheduler.step(args.start_epoch)

    # There are 4 options for evaluation:
    #  --eval val                           just run validation
    #  --eval val --dump_assets             dump all images and assets
    #  --eval folder                        just dump all basic images
    #  --eval folder --dump_assets          dump all images and assets
    if args.eval == 'val':

        if args.dump_topn:
            validate_topn(val_loader, net, criterion_val, optim, 0, args)
        else:
            validate(val_loader, net, criterion=criterion_val, optim=optim, epoch=0,
                     dump_assets=args.dump_assets,
                     dump_all_images=args.dump_all_images,
                     calc_metrics=not args.no_metrics)
        return 0
    elif args.eval == 'folder':
        # Using a folder for evaluation means to not calculate metrics
        validate(val_loader, net, criterion=None, optim=None, epoch=0,
                 calc_metrics=False, dump_assets=args.dump_assets,
                 dump_all_images=True)
        return 0
    elif args.eval is not None:
        raise 'unknown eval option {}'.format(args.eval)

    #--------------
    #GAN
    #adversarial_loss = torch.nn.BCELoss()   #GAN
    adversarial_loss = torch.nn.BCEWithLogitsLoss()     #amp는 BCELoss 작동안함.



    for epoch in range(args.start_epoch, args.max_epoch):
        run["val/Learning rate"].log(optimizer_D.param_groups[0]['lr'])
        update_epoch(epoch)

        if args.only_coarse:
            train_obj.only_coarse()
            train_obj.build_epoch()
            if args.apex:
                train_loader.sampler.set_num_samples()

        elif args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.disable_coarse()
                train_obj.build_epoch()
                if args.apex:
                    train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()
        else:
            pass

        train(train_loader, net, optim, epoch,discriminator,optimizer_D,adversarial_loss)

        if args.apex:
            train_loader.sampler.set_epoch(epoch + 1)

        if epoch % args.val_freq == 0:
            validate(val_loader, net, criterion_val, optim, epoch,discriminator,optimizer_D)

        visualize(val_loader, net, criterion_val, optim, epoch,discriminator,optimizer_D)       
        
        scheduler.step()
          

        if check_termination(epoch):
            return 0

    run.stop()      #stop neptune


def train(train_loader, segmentation, optimizer_S, curr_epoch,discriminator,optimizer_D,adversarial_loss):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer_S: optimizer of semantic segmentation
    curr_epoch: current epoch
    return:

    #Jacob
    discriminator: Discriminator for GAN
    optimizer_D: Optimizer of discriminator
    """

    train_stage1_loss = AverageMeter()
    train_stage2_loss = AverageMeter()
    start_time = None
    warmup_iter = 10

    for i, data in enumerate(train_loader):
        if i <= warmup_iter:
            start_time = time.time()

        #Loading images
        # inputs = (bs,3,713,713)
        # gts    = (bs,713,713)
        images, gts, _img_name, scale_float = data
        batch_pixel_size = images.size(0) * images.size(2) * images.size(3)
        images, gts, scale_float = images.cuda(), gts.cuda(), scale_float.cuda()
                
        inputs = {'images': images, 'gts': gts}

        real_labels = torch.ones(train_loader.batch_size, 1).cuda()
        fake_labels = torch.zeros(train_loader.batch_size, 1).cuda()

        segmentation.train()
        discriminator.train()

        #------------------
        #   Stage 1
        #   Training Segmentation model with Seg loss and Adv loss (BCE loss)
        #------------------
        optimizer_S.zero_grad()

        seg_loss,out = segmentation(inputs)     #seg_loss는 CrossEntropyLoss이고, out은 Segmentation 마스크임.
        #   'out' the network prediction, shape (batch,19,H,W)

        smax_out = torch.nn.functional.softmax(out, dim=1)  #(batch,19,H,W)

        #-----------------
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        # The Cityscapes dataset has origianlly 34 classes. But when you use this for training, You should convert 34 classes to 19 classes.
        # And When you train your model, There must be TrainID 255. That is, 'unlabeled, poleground,caravan, trailer, out or roi, etc..'.
        # ignore those 255 IDs.
        #-----------------       

        adv_loss = adversarial_loss(discriminator(smax_out),real_labels)
        stage1_loss = seg_loss + adv_loss

        #---
        if args.apex:
            log_stage1_loss = stage1_loss.clone().detach_()
            torch.distributed.all_reduce(log_stage1_loss,
                                         torch.distributed.ReduceOp.SUM)
            log_stage1_loss = log_stage1_loss / args.world_size
        else:
            stage1_loss = stage1_loss.mean()
            log_stage1_loss = stage1_loss.clone().detach_()

        train_stage1_loss.update(log_stage1_loss.item(), batch_pixel_size)        #??? 이게 뭐지?
        if args.fp16:
            with amp.scale_loss(stage1_loss, optimizer_S) as scaled_loss:
                scaled_loss.backward()
        else:
            stage1_loss.backward()
        #---
        optimizer_S.step()

        #------------------
        #   Stage 2
        #   Training Discriminator ONLY
        #------------------
        optimizer_D.zero_grad()   

        onehot_gts=torch.nn.functional.one_hot(gts,256)                 #(batch,H,W,256)
        onehot_gts=torch.permute(onehot_gts,(0,3,1,2))[:,0:19]          #(batch,256,H,W)   torch.int64  Note. permute and transpose are Not same. It can look similar but not same. 0~18 and 255 dimension contains 1.
        gts_float = onehot_gts.detach().clone().type(torch.float32)     #(batch,19,H,W)    torch.float32

        real_loss = adversarial_loss(discriminator(gts_float),real_labels)          #문제 없지않을까?
        fake_loss = adversarial_loss(discriminator(smax_out.detach()),fake_labels)

        stage2_loss = real_loss + fake_loss

        #---
        if args.apex:
            log_stage2_loss = stage2_loss.clone().detach_()
            torch.distributed.all_reduce(log_stage2_loss,
                                        torch.distributed.ReduceOp.SUM)
            log_stage2_loss = log_stage2_loss / args.world_size
        else:
            stage2_loss = stage2_loss.mean()
            log_stage2_loss = stage2_loss.clone().detach_()

        train_stage2_loss.update(log_stage2_loss.item(), batch_pixel_size)
        if args.fp16:
            with amp.scale_loss(stage2_loss, optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            stage2_loss.backward()

        #---
        optimizer_D.step()


        if i >= warmup_iter:
            curr_time = time.time()
            batches = i - warmup_iter + 1
            batchtime = (curr_time - start_time) / batches
        else:
            batchtime = 0

        msg = ('[epoch {}], [iter {} / {}], [train stage1 loss {:0.6f} (Seg loss {}, Adv loss {})], [Stage1 lr {:0.6f}]'
               '\n[train stage2 loss {:0.6f} (real loss {}, fake loss {})],'
               ' [Stage2 lr {:0.6f}] [batchtime {:0.3g}]')
        msg = msg.format(
            curr_epoch, i + 1, len(train_loader), train_stage1_loss.avg,seg_loss,adv_loss,optimizer_S.param_groups[-1]['lr'],
            train_stage2_loss.avg,real_loss,fake_loss,optimizer_D.param_groups[-1]['lr'], batchtime)
        logx.msg(msg)

        metrics = {'Seg loss': train_stage1_loss.avg,
                   'lr': optimizer_S.param_groups[-1]['lr']}
        curr_iter = curr_epoch * len(train_loader) + i
        logx.metric('train', metrics, curr_iter)

        #netune
        run["train/Stage1 loss"].log(train_stage1_loss.avg)
        run["train/Seg loss"].log(seg_loss)
        run["train/Adv loss"].log(adv_loss)

        run["train/Stage2 loss"].log(train_stage2_loss.avg)
        run["train/Real loss"].log(real_loss)
        run["train/Fake loss"].log(fake_loss)

        run["train/Learning rate"].log(optimizer_D.param_groups[0]['lr'])

        if i >= 10 and args.test_mode:
            del data, inputs, gts
            return
        del data
        torch.cuda.empty_cache()



def validate(val_loader, net, criterion, optim, epoch,
             discriminator,optimizer_D,
             calc_metrics=True,
             dump_assets=False,
             dump_all_images=False):
    """
    Run validation for one epoch

    :val_loader: data loader for validation
    :net: the network
    :criterion: loss fn
    :optimizer: optimizer
    :epoch: current epoch
    :calc_metrics: calculate validation score
    :dump_assets: dump attention prediction(s) images
    :dump_all_images: dump all images, not just N
    """
    dumper = ImageDumper(val_len=len(val_loader),
                         dump_all_images=dump_all_images,
                         dump_assets=dump_assets,
                         dump_for_auto_labelling=args.dump_for_auto_labelling,
                         dump_for_submission=args.dump_for_submission)

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0

    for val_idx, data in enumerate(val_loader):
        input_images, labels, img_names, _ = data 
        if args.dump_for_auto_labelling or args.dump_for_submission:
            submit_fn = '{}.png'.format(img_names[0])
            if val_idx % 20 == 0:
                logx.msg(f'validating[Iter: {val_idx + 1} / {len(val_loader)}]')
            if os.path.exists(os.path.join(dumper.save_dir, submit_fn)):
                continue

        # Run network
        assets, _iou_acc = \
            eval_minibatch(data, net, criterion, val_loss, calc_metrics,
                          args, val_idx)

        iou_acc += _iou_acc

        input_images, labels, img_names, _ = data

        # dumper.dump({'gt_images': labels,
        #              'input_images': input_images,
        #              'img_names': img_names,
        #              'assets': assets}, val_idx)

        if val_idx > 5 and args.test_mode:
            break

        if val_idx % 20 == 0:
            logx.msg(f'validating[Iter: {val_idx + 1} / {len(val_loader)}]')

    was_best = False
    if calc_metrics:
        was_best = eval_metrics(iou_acc, args, net, optim, val_loss, epoch,discriminator=discriminator,optimizer_D=optimizer_D,Neptune_run=run)

    # Write out a summary html page and tensorboard image table
    if not args.dump_for_auto_labelling and not args.dump_for_submission:
        dumper.write_summaries(was_best)
        
def visualize(val_loader, net, criterion, optim, epoch,
             discriminator,optimizer_D,
             calc_metrics=True,
             dump_assets=False,
             dump_all_images=False):
    """
    Run validation for one epoch

    :val_loader: data loader for validation
    :net: the network
    :criterion: loss fn
    :optimizer: optimizer
    :epoch: current epoch
    :calc_metrics: calculate validation score
    :dump_assets: dump attention prediction(s) images
    :dump_all_images: dump all images, not just N
    """
    dumper = ImageDumper(val_len=len(val_loader),
                         dump_all_images=dump_all_images,
                         dump_assets=dump_assets,
                         dump_for_auto_labelling=args.dump_for_auto_labelling,
                         dump_for_submission=args.dump_for_submission)

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0

    for val_idx, data in enumerate(val_loader):
        input_images, labels, img_names, _ = data 
        if args.dump_for_auto_labelling or args.dump_for_submission:
            submit_fn = '{}_{}.png'.format(img_names[0],epoch)
            if val_idx % 20 == 0:
                logx.msg(f'validating[Iter: {val_idx + 1} / {len(val_loader)}]')
            if os.path.exists(os.path.join(dumper.save_dir, submit_fn)):
                continue

        # Run network
        assets, _iou_acc = \
            eval_minibatch(data, net, criterion, val_loss, calc_metrics,
                          args, val_idx)

        iou_acc += _iou_acc

        input_images, labels, img_names, _ = data

        dumper.dump({'gt_images': labels,
                     'input_images': input_images,
                     'img_names': img_names,
                     'assets': assets}, val_idx,epoch)
        
        break
    
if __name__ == '__main__':
    main()
