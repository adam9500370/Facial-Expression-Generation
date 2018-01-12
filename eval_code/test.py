import time

import torch
import numpy as np

from evaluation import AverageMeter, accuracy, get_inception_score, build_confusion_mtx


def test(test_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    #preds = np.zeros((0,7,))
    pred_labels = np.zeros([0,])
    GT_labels = np.zeros([0,])
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        '''
        cal_probs = torch.nn.Softmax(dim=0)
        probs = cal_probs(output)
        preds = np.concatenate([preds, probs.data.cpu().numpy()], axis=0)
        '''

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        _, pred = output.data.topk(1, 1, True, True)
        pred_labels = np.concatenate([pred_labels, pred.cpu().numpy().flatten()], axis=0)
        GT_labels = np.concatenate([GT_labels, target.cpu().numpy().flatten()], axis=0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    build_confusion_mtx(GT_labels, pred_labels, categories)

    '''
    mean_score, std_score = get_inception_score(preds)
    print(' * IS: mean {mean_score:.3f} std {std_score:.3f}'.format(mean_score=mean_score, std_score=std_score))
    '''

    return top1.avg
