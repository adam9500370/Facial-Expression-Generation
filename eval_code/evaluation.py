import os
import itertools

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results


def get_inception_score(preds, num_splits=10):
    scores = []
    for i in range(num_splits):
        part = preds[(i * preds.shape[0] // num_splits):((i+1) * preds.shape[0] // num_splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def build_confusion_mtx(GT_labels, pred_labels, abbr_categories, checkpoints_path='checkpoints'):
    # Compute confusion matrix
    cm = confusion_matrix(GT_labels, pred_labels)
    np.set_printoptions(precision=2)
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(figsize=(16,16))
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')

    #plt.show()

    fig.savefig(os.path.join(checkpoints_path, 'confusion_matrix.png'))
    plt.close(fig)
     
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45, fontsize=20)
    plt.yticks(tick_marks, category, fontsize=20)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] >= 0.5:
            plt.text(j, i, np.around(cm[i, j], decimals=2), horizontalalignment="center", color="white", fontsize=20)
        else:
            plt.text(j, i, np.around(cm[i, j], decimals=2), horizontalalignment="center", color="black", fontsize=20)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
