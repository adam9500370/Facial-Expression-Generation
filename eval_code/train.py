import os
import shutil
import time

import torch

from evaluation import AverageMeter, accuracy


def train(train_loader, model, criterion, optimizer, epoch, print_freq, summary_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    summary_writer.add_scalar('data/losses_avg', losses.avg, epoch)
    summary_writer.add_scalar('data/top1_avg', top1.avg, epoch)
    summary_writer.add_scalar('data/top5_avg', top5.avg, epoch)


def save_checkpoint(state, is_best, checkpoints_path='checkpoints'):
    filename = os.path.join(checkpoints_path, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_model_filename = os.path.join(checkpoints_path, 'model_best.pth.tar')
        shutil.copyfile(filename, best_model_filename)


def adjust_learning_rate(orig_lr, optimizer, epoch, lr_decay, lr_decay_freq):
    """Sets the learning rate to the initial LR decayed by {lr_decay} every {lr_decay_freq} epochs"""
    lr = orig_lr * (lr_decay ** (epoch // lr_decay_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
