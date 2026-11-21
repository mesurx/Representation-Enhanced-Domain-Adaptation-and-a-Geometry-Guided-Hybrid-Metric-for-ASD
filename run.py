import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from net_ablation_MRFF import STgramMFN
from trainer_ablation_MRFF import Trainer
from dataset import ASDDataset,BalancedDomainBatchSampler
import utils

sep = os.sep

def main(args):
    # set random seed
    utils.setup_seed(args.random_seed)
    args.dp = False
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dirs = args.train_dirs  #  + args.add_dirs
    args.meta2label, _,args.machine_label,_ = utils.metadata_to_label(train_dirs) # 字典

    train_file_list = []
    for train_dir in train_dirs:
        train_file_list.extend(utils.get_filename_list(train_dir))
    train_dataset = ASDDataset(args, train_file_list, load_in_memory=False) # 总的样本数量为41999
    balanced_batch_sampler = BalancedDomainBatchSampler(file_list=train_file_list,batch_size=32,source_ratio=0.75,drop_last=True)
    train_dataloader = DataLoader(train_dataset,batch_sampler=balanced_batch_sampler,num_workers=args.num_workers)

    # set model
    args.num_classes = len(args.meta2label.keys())
    args.logger.info(f'Num classes: {args.num_classes}')
    net = STgramMFN(num_classes=args.num_classes,num_domains=2, use_arcface=args.use_arcface,
                    m=float(args.m), s=float(args.s), sub=args.sub_center)
    net = net.to(args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.1*float(args.lr))

    trainer = Trainer(args=args,
                      net=net,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      transform=train_dataset.transform,)
    # train model
    if not args.load_epoch:
        trainer.train(train_dataloader)

def run():
    # init config parameters
    params = utils.load_yaml(file_path='./config.yaml')
    parser = argparse.ArgumentParser(description=params['description'])
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=utils.set_type)
    args = parser.parse_args()
    # init logger and writer
    time_str = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
    args.version = f'STgram-MFN(m={args.m},s={args.s})'
    args.version = f'{time_str}-{args.version}' if not args.load_epoch and args.time_version else args.version
    log_dir = f'runs/{args.version}'
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))
    # save version files
    if args.save_version_files: utils.save_load_version_files(log_dir, args.save_version_file_patterns, args.pass_dirs)
    # run
    args.writer, args.logger = writer, logger
    args.logger.info(args.version)
    main(args)

if __name__ == '__main__':
    run()