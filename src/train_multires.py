# This script is for multi-resolution training, including original FairMOT, and DeepScale
# Author: Renjie Xu
# Time: 2023/3/23
# Command: # python train_multires.py --task mot --exp_id exp_name --data_cfg '../src/lib/cfg/mot17_half.json' --load_model ./model_best.pth --batch_size 64 --num_epochs 500 --gpus 0,1,2,3 --arch half-dla_34 --num_workers 16 --lr_step 400,450 --multi_res_train True

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib.pyplot import pause

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory



def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)         
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)                    
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':                                     # load pretrained model
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)
    if opt.multi_res_train:
        resolutions = [(576, 320),(640, 352),(704, 384),(864, 480),(1088, 608)]
        res_weights = [0.2, 0.4, 0.6, 0.8, 1.0]
        # res_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        resolutions = [(1088, 608)]
        res_weights = [1.0]

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        for idx, res in enumerate(resolutions):
            print('Setting resolution: ', res)
            dataset = Dataset(opt, dataset_root, trainset_paths, res, augment=True, transforms=transforms)
            opt = opts().update_dataset_info_and_set_heads(opt, dataset)
            train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True)

            # continue
            log_dict_train, _ = trainer.train(epoch, train_loader, res_weights[idx])
            logger.write('epoch: {} |'.format(epoch))
            for k, v in log_dict_train.items():
                logger.scalar_summary('train_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                    epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                    epoch, model, optimizer)
            save_model(os.path.join(opt.save_dir, 'model_IDClassifier_last.pth'),
                       epoch, trainer.loss.classifier)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                    epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch % 5 == 0 or epoch >= 2:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                    epoch, model, optimizer)
            save_model(os.path.join(opt.save_dir, 'model_IDClassifier_{}.pth'.format(epoch)),
                        epoch, trainer.loss.classifier)
    logger.close()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    opt = opts().parse()
    main(opt)

