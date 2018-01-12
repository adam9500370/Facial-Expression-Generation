import argparse
import os

import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tensorboardX import SummaryWriter

from dataloader import read_fer2013_data
from model import create_model, load_model_from_checkpoint
from train import train, save_checkpoint, adjust_learning_rate
from test import test

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch FER2013 Training')
parser.add_argument('--data', metavar='DIR', default=os.path.join('..', 'data', 'fer2013'), type=str,
                    help='path to dataset (default: ../data/fer2013)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--num-classes', default=7, type=int, metavar='N',
                    help='number of classes (default: 7)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='LRD', help='learning rate decay (default: 0.1)')
parser.add_argument('--lr-decay-freq', default=20, type=int,
                    metavar='N', help='learning rate decay frequency (default: 20)')
parser.add_argument('--print-freq', '-p', default=300, type=int,
                    metavar='N', help='print frequency (default: 300)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model (on ImageNet)')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='fine tune pre-trained model')

# directory for saving intermediate checkpoints
checkpoints_path = 'checkpoints'
if not os.path.exists(checkpoints_path):
    os.mkdir(checkpoints_path)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    model = create_model(args.arch, args.pretrained, args.finetune, num_classes=args.num_classes)

    # define loss function (criterion) and optimizer
    criterion = CrossEntropyLoss().cuda()

    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                    args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            load_model_from_checkpoint(args, model, optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # data loading
    train_path = os.path.join(args.data, 'train')
    test_path = os.path.join(args.data, 'test')
    if os.path.exists(train_path):
        train_loader = read_fer2013_data(train_path, dataset_type='train', batch_size=args.batch_size, num_workers=args.workers)
    if os.path.exists(test_path):
        test_loader = read_fer2013_data(test_path, dataset_type='test', batch_size=args.batch_size, num_workers=args.workers)

    if args.evaluate:
        test(test_loader, model, criterion, args.print_freq)
        return

    summary_writer = SummaryWriter()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args.lr, optimizer, epoch, args.lr_decay, args.lr_decay_freq)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq, summary_writer)

        # evaluate on test set
        prec1 = test(test_loader, model, criterion, args.print_freq)

        # remember best prec@1 and save all checkpoints
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    summary_writer.close()



if __name__ == '__main__':
    main()
