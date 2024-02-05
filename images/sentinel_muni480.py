# %% [markdown]
# ### import time
import torch.utils.data
import os
import sys
import rasterio
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import re

from math import cos,pi
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, accuracy_score, confusion_matrix
from scipy.ndimage import morphology
from scipy.ndimage.filters import maximum_filter1d
from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Dropout3d
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')

# %%
!mkdir /kaggle/working/

# %%
root_path = '/kaggle/input/sentinel2-munich480/munich480'  # root dir dataset
result_path = '/kaggle/working/'
resume_path = '/kaggle/input/pretrained-deeplab/best_modelDeepLab.pth'
result_train = 'train_resultsDeepLab.txt'
result_validation = 'validation_resultsDeepLab.txt'
no_of_classes = 18 #5
workers = 8
batch_size = 2
h = w = 7
n_classes = 18 #400
LABEL_FILENAME = "y.tif"
best_test_acc = 0
loss = 'batch'
ottimizzatore = 'sgd'
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9
loss_weights = 'store_true'
ignore_index = 0
test_only = False
sample_duration = 30
n_epochs = 1

num_folds = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings("ignore")

# %%
def save(path, model, optimizer, **kwargs):
    model_state = None
    optimizer_state = None
    if model is not None:
        model_state = model.state_dict()
    if optimizer is not None:
        optimizer_state = optimizer.state_dict()
    torch.save(
        dict(model_state=model_state,
             optimizer_state=optimizer_state,
             **kwargs),
        path
    )


def resume(path, model, optimizer):
    if torch.cuda.is_available():
        snapshot = torch.load(path)
    else:
        snapshot = torch.load(path, map_location="cpu")
    print("Loaded snapshot from", path)

    model_state = snapshot.pop('model_state', snapshot)
    optimizer_state = snapshot.pop('optimizer_state', None)

    if model is not None and model_state is not None:
        print("loading model...")
        model.load_state_dict(model_state)

    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return snapshot

# %%
LABEL_FILENAME = "y.tif"
class SentinelDataset(torch.utils.data.Dataset):
    '''
    If the first label is for example "1|unknown" then this will be replaced with a 0 (zero).
    If you want to ignore other labels, then remove them from the classes.txt file and
    this class will assigne label 0 (zero).
    Warning: this tecnique is not stable!
    '''

    def __init__(self, root_dir, seqlength=30, tileids=None):
        self.root_dir = root_dir
        self.name = os.path.basename(root_dir)
        self.data_dirs = [d for d in os.listdir(self.root_dir) if d.startswith("data")]
        self.seqlength = seqlength
        self.munich_format = None
        self.src_labels = None
        self.dst_labels = None
        # labels read from groudtruth files (y.tif)
        # useful field to check the available labels
        self.unique_labels = np.array([], dtype=float)

        self.b8_index = 3  # munich dataset
        self.b4_index = 2  # munich dataset

        stats = dict(
            rejected_nopath=0,
            rejected_length=0,
            total_samples=0)

        # statistics
        self.samples = list()

        self.ndates = list()

        dirs = []
        if tileids is None:
            # files = os.listdir(self.data_dirs)
            for d in self.data_dirs:
                dirs_name = os.listdir(os.path.join(self.root_dir, d))
                dirs_path = [os.path.join(self.root_dir, d, f) for f in dirs_name]
                dirs.extend(dirs_path)
        else:
            # tileids e.g. "tileids/train_fold0.tileids" path of line separated tileids specifying
            with open(os.path.join(self.root_dir, tileids), 'r') as f:
                files = [el.replace("\n", "") for el in f.readlines()]
            for d in self.data_dirs:
                dirs_path = [os.path.join(self.root_dir, d, f) for f in files]
                dirs.extend(dirs_path)

        self.classids, self.classes = self.read_classes(os.path.join(self.root_dir, "classes.txt"))

        for path in dirs:
            if not os.path.exists(path):
                stats["rejected_nopath"] += 1
                continue
            if not os.path.exists(os.path.join(path, LABEL_FILENAME)):
                stats["rejected_nopath"] += 1
                continue

            ndates = len(get_dates(path))

            if ndates < self.seqlength:
                stats["rejected_length"] += 1
                continue  # skip shorter sequence lengths

            stats["total_samples"] += 1
            self.samples.append(path)
            self.ndates.append(ndates)

        print_stats(stats)

    def read_classes(self, csv):
        with open(csv, 'r') as f:
            classes = f.readlines()

        ids = list()
        names = list()
        for row in classes:
            row = row.replace("\n", "")
            if '|' in row:
                id, cl = row.split('|')
                ids.append(int(id))
                names.append(cl)

        return ids, names

    def get_image_h_w(self):
        label, profile = read(os.path.join(self.samples[0], LABEL_FILENAME))
        return label.shape[-2], label.shape[-1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # path = os.path.join(self.data_dir, self.samples[idx])
        path = self.samples[idx]
        if path.endswith(os.sep):
            path = path[:-1]
        patch_id = os.path.basename(path)

        label, profile = read(os.path.join(path, LABEL_FILENAME))

        profile["name"] = self.samples[idx]

        # unique dates sorted ascending
        dates = get_dates(path, n=self.seqlength)

        x10 = list()
        x20 = list()
        x60 = list()

        for date in dates:
            if self.munich_format is None:
                self.munich_format = os.path.exists(os.path.join(path, date + "_10m.tif"))
                if self.munich_format:  # munich dataset
                    self.b8_index = 3
                    self.b4_index = 2
                else:  # IREA dataset
                    self.b8_index = 6
                    self.b4_index = 2
            if self.munich_format:
                x10.append(read(os.path.join(path, date + "_10m.tif"))[0])
                x20.append(read(os.path.join(path, date + "_20m.tif"))[0])
                x60.append(read(os.path.join(path, date + "_60m.tif"))[0])
            else:
                x10.append(read(os.path.join(path, date + ".tif"))[0])

        x10 = np.array(x10) * 1e-4
        if self.munich_format:
            x20 = np.array(x20) * 1e-4
            x60 = np.array(x60) * 1e-4

        # augmentation
        # if np.random.rand() < self.augmentrate:
        #     x10 = np.fliplr(x10)
        #     x20 = np.fliplr(x20)
        #     x60 = np.fliplr(x60)
        #     label = np.fliplr(label)
        # if np.random.rand() < self.augmentrate:
        #     x10 = np.flipud(x10)
        #     x20 = np.flipud(x20)
        #     x60 = np.flipud(x60)
        #     label = np.flipud(label)
        # if np.random.rand() < self.augmentrate:
        #     angle = np.random.choice([1, 2, 3])
        #     x10 = np.rot90(x10, angle, axes=(2, 3))
        #     x20 = np.rot90(x20, angle, axes=(2, 3))
        #     x60 = np.rot90(x60, angle, axes=(2, 3))
        #     label = np.rot90(label, angle, axes=(0, 1))

        # replace stored ids with index in classes csv
        label = label[0]
        self.unique_labels = np.unique(np.concatenate([label.flatten(), self.unique_labels]))
        new = np.zeros(label.shape, np.int)
        for cl, i in zip(self.classids, range(len(self.classids))):
            new[label == cl] = i

        label = new

        label = torch.from_numpy(label)
        x10 = torch.from_numpy(x10)
        if self.munich_format:
            x20 = torch.from_numpy(x20)
            x60 = torch.from_numpy(x60)

            x20 = F.interpolate(x20, size=x10.shape[2:4])
            x60 = F.interpolate(x60, size=x10.shape[2:4])

            x = torch.cat((x10, x20, x60), 1)
        else:
            x = x10

        # permute channels with time_series (t x c x h x w) -> (c x t x h x w)
        x = x.permute(1, 0, 2, 3)

        x = x.float()
        label = label.long()

        target_ndvi = get_all_signatures(x, label, len(self.classids), self.b4_index, self.b8_index)

        return x, label, target_ndvi.float(), dates, patch_id


def get_all_signatures(inp, target, num_cls, b4_index, b8_index):
    """
    expected input having shape  (c, t, h, w) and target (h, w)
        c = number of channels for each sentinel-2 image
        t = number of images in the time series
        hxw = image size
    """
    c, t, h, w = inp.shape
    output_ndvi = np.zeros((t, h, w), dtype=np.float)

    # xin = torch.linspace(1, t, t)

    for cls_index_ in range(0, num_cls):
        pts = (target == cls_index_).numpy()
        all_ndvi_x_cls = []
        for row, yr in enumerate(pts):
            for col, xc in enumerate(yr):
                if xc:  # is True
                    # if target[batch_index_, row, col].item() != cls_index_:
                    #     print("error")
                    b8 = inp[b8_index, :, row, col]
                    b4 = inp[b4_index, :, row, col]
                    ndvi = (b8 - b4) / (b8 + b4)
                    ndvi = np.nan_to_num(ndvi.numpy())
                    # if np.isnan(ndvi).any():
                    #     print("NAN in ndvi!")
                    all_ndvi_x_cls.append(ndvi)
        mean_ndvi = np.zeros((t,), dtype=float)
        if len(all_ndvi_x_cls) > 1:
            mean_ndvi = np.mean(all_ndvi_x_cls, axis=0)
        if len(all_ndvi_x_cls) == 1:
            mean_ndvi = all_ndvi_x_cls[0]
        mmax_ndvi = __max_filter1d_valid(mean_ndvi, 5)  # moving max x class

        # print("batch", batch_index_, ", cls", cls_index_, ", ndvi", mmax_ndvi)
        # plt.plot(xin, mmax_ndvi)

        output_ndvi[:, pts] = mmax_ndvi.reshape(t, 1)
    # plt.show()
    return torch.from_numpy(output_ndvi).float()


def __max_filter1d_valid(a, w):
    b = a.clip(min=0)  # transform negative elements to zero
    return maximum_filter1d(b, size=w)


def read(file):
    with rasterio.open(file) as src:
        return src.read(), src.profile


def get_dates(path, n=None):
    """
    extracts a list of unique dates from dataset sample

    :param path: to dataset sample folder
    :param n: choose n random samples from all available dates
    :return: list of unique dates in YYYYMMDD format
    """

    files = os.listdir(path)
    dates = list()
    for f in files:
        f = f.split("_")[0]
        if len(f) == 8:  # 20160101
            dates.append(f)

    dates = set(dates)

    if n is not None:
        dates = random.sample(dates, n)

    dates = list(dates)
    dates.sort()
    return dates


def print_stats(stats):
    print_lst = list()
    for k, v in zip(stats.keys(), stats.values()):
        print_lst.append("{}:{}".format(k, v))
    print('\n', ", ".join(print_lst))


# %%
print("Starting loading Dataset...")

traindataset = SentinelDataset(root_path, tileids="tileids/train_fold0.tileids", seqlength=sample_duration)
traindataloader = torch.utils.data.DataLoader(
    traindataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last = True)
# How to iterate on a dataloader
for iteration, data in enumerate(traindataloader):
    input, target, target_ndvi, _, _ = data
    print('input temporal series with 30 images of size 13x48x48:', input.shape)
    print('target segmentation image (batchx48x48):', target.shape)
    print('target_ndvi containing 30 channels of size 48x48:', target_ndvi.shape)
    break

# Load test set
testdataset = SentinelDataset(root_path, tileids="tileids/test_fold0.tileids", seqlength=sample_duration)
testdataloader = torch.utils.data.DataLoader(
    testdataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last = True)
# Load validation set
validationdataset = SentinelDataset(root_path, tileids="tileids/eval.tileids", seqlength=sample_duration)
validationdataloader = torch.utils.data.DataLoader(
    validationdataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last = True)

numclasses = len(traindataset.classes)
labels = list(range(numclasses))

# %%
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


# %%
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=-100, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.criterion_xent = None
        self.criterion_mse = None
        self.beta = 0.9999  # for class balanced loss

    def build_loss(self, mode='ce'):
        loss_func = None
        """Choices: ['ce' | 'focal' | 'ndvi' | 'batch']"""
        if mode == 'ce':
            self.criterion_xent = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
            loss_func = self.cross_entropy_loss
        elif mode == 'focal':
            self.criterion_xent = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
            loss_func = self.focal_loss
        elif mode == 'ndvi':
            self.criterion_xent = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
            self.criterion_mse = nn.MSELoss()  # Default reduction: 'mean' (reduction='sum')
            # self.criterion_mse = nn.L1Loss()
            loss_func = self.ndvi_loss
        elif mode == 'batch':
            # weights are computed inside a batch:
            # https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
            self.criterion_xent = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
            loss_func = self.class_balanced_loss
        else:
            raise NotImplementedError

        if self.cuda:
            self.criterion_xent = self.criterion_xent.cuda()
            if self.criterion_mse is not None:
                self.criterion_mse = self.criterion_mse.cuda()

        return loss_func

    def cross_entropy_loss(self, logit, target):
        n, c, h, w = logit.size()

        loss = self.criterion_xent(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def focal_loss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()

        logpt = -self.criterion_xent(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def ndvi_loss(self, ndvi_features, logit, target_cls, ndvi_target, samples_per_cls, weight_mse=1.0):
        n, c, h, w = logit.size()
        assert (not torch.isnan(ndvi_target).any())

        # Effective Number of Samples (ENS)
        if not samples_per_cls == None:
            assert (samples_per_cls.shape[0] == c)
            weights = (1.0 - self.beta) / (1.0 - torch.pow(self.beta, samples_per_cls.float()))
            weights[weights == float('inf')] = 0
            weights = weights / torch.sum(weights) * c  # wights in the range [0, c]
            self.criterion_xent.weight = weights

        loss_xent = self.criterion_xent(logit, target_cls.long())
        loss_mse = self.criterion_mse(ndvi_features, ndvi_target)
        loss_mse *= weight_mse
        loss = loss_xent + loss_mse

        if self.batch_average:
            loss /= n

        return loss

    def class_balanced_loss(self, logit, target, samples_per_cls, weight_type='ENS'):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks."""
        # Starting point:
        # https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
        n, c, h, w = logit.size()
        # beta = (self.total_num_samples - 1) / self.total_num_samples

        assert(samples_per_cls.shape[0] == c)

        # compute weights for each minibatch
        if weight_type == 'ENS':
            # Effective Number of Samples (ENS)
            weights = (1.0 - self.beta) / (1.0 - torch.pow(self.beta, samples_per_cls.float()))
            weights[weights == float('inf')] = 0
        elif weight_type == 'ISNS':
            # Inverse of Square Root of Number of Samples (ISNS)
            weights = 1.0 / torch.sqrt(torch.tensor([2, 1000, 1, 20000, 500]).float())
        else:
            # Inverse of Number of Samples (INS)
            weights = 1.0 / torch.tensor([2, 1000, 1, 20000, 500]).float()

        weights = weights / torch.sum(weights) * c  # wights in the range [0, c]

        self.criterion_xent.weight = weights
        loss = self.criterion_xent(logit, target.long())
        # print("loss weights:", self.criterion_xent.weight)

        if self.batch_average:
            loss /= n

        return loss
    


# %%
class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next restult to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


# %%
class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    pass

class ConfusionMatrix(Metric):

    def __init__(self, labels, ignore_class=None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_class = ignore_class  # the class index to be removed
        self.labels = labels
        self.n_classes = len(self.labels)
        if self.ignore_class is not None:
            self.matrix = np.zeros((self.n_classes-1, self.n_classes-1))
        else:
            self.matrix = np.zeros((self.n_classes, self.n_classes))

        # self.pred = []
        # self.targ = []

    def get_labels(self):
        if self.ignore_class is not None:
            return np.delete(self.labels, self.ignore_class)
        return self.labels

    def forward(self, y_pr, y_gt):
        # sklearn.metrics
        pred = y_pr.view(-1).cpu().detach().tolist()
        targ = y_gt.view(-1).cpu().detach().tolist()

        # To format the matrix
        # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
        # confusion_matrix(y_true, y_pred)
        # array([[2, 0, 0],  # two zeros were predicted as zeros
        #        [0, 0, 1],  # one 1 was predicted as 2
        #        [1, 0, 2]])  # two 2s were predicted as 2, and one 2 was 0
        matrix = confusion_matrix(targ, pred, labels=self.labels)

        if self.ignore_class is not None:
            matrix = np.delete(matrix, self.ignore_class, 0)  # remove the row
            matrix = np.delete(matrix, self.ignore_class, 1)  # remove the column

        self.matrix = np.add(self.matrix, matrix)

        results_vec = {"labels": self.get_labels(), "confusion matrix": self.matrix}

        total = np.sum(self.matrix)
        true_positive = np.diag(self.matrix)
        sum_rows = np.sum(self.matrix, axis=0)
        sum_cols = np.sum(self.matrix, axis=1)
        false_positive = sum_rows - true_positive
        false_negative = sum_cols - true_positive
        # calculate accuracy
        overall_accuracy = np.sum(true_positive) / total
        results_scalar = {"OA": overall_accuracy}

        # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
        p0 = np.sum(true_positive) / total
        pc = np.sum(sum_rows * sum_cols) / total ** 2
        kappa = (p0 - pc) / (1 - pc)
        results_scalar["Kappa"] = kappa

        # Per class recall, prec and F1
        recall = true_positive / (sum_cols + 1e-12)
        results_vec["R"] = recall
        precision = true_positive / (sum_rows + 1e-12)
        results_vec["P"] = precision
        f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)
        results_vec["F1"] = f1

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
            results_vec["IoU"] = iou
            results_scalar["mIoU"] = np.nanmean(iou)

        # Per class accuracy
        cl_acc = true_positive / (sum_cols + 1e-12)
        results_vec["Acc"] = cl_acc

        # weighted measures
        prob_c = sum_rows / total
        prob_r = sum_cols / total
        recall_weighted = np.inner(recall, prob_r)
        results_scalar["wR"] = recall_weighted
        precision_weighted = np.inner(precision, prob_r)
        results_scalar["wP"] = precision_weighted
        f1_weighted = 2 * (recall_weighted * precision_weighted) / (recall_weighted + precision_weighted)
        results_scalar["wF1"] = f1_weighted
        random_accuracy = np.inner(prob_c, prob_r)
        results_scalar["RAcc"] = random_accuracy

        return results_vec, results_scalar

# %%
def write_signatures(fcn, target, output, target_ndvi, out_ndvi, dates, patch_id, set_name):
    # target_ndvi = torch.randint(0, nc, size=(b, s, h, w))
    # patch_id = ('11', '12')
    # dates = [('20200101', '20200301'), ('20200102', '20200302'), ('20200103', '20200303'), ('20200104', '20200304'),
    #          ('20200105', '20200305')]
    nb, nt, h, w = target_ndvi.shape
    assert(len(dates) == nt)
    assert(len(dates[0]) == nb)

    # winner class for each pixel
    winners = torch.softmax(output, dim=1).argmax(dim=1)

    # bias = fcn.conv5out.bias
    weight = fcn.conv5out.weight

    for idx, patch_name in enumerate(patch_id):
        # print('patch_name:', patch_name)
        with open(os.path.join(result_path, set_name + '_patch_' + patch_name + ".txt"), 'w') as f:
            f.write('class, output, type_ndvi, x, y, ')
            f.write(', '.join(map(str, [e[idx] for e in dates])))
            f.write("\n")
            for y in range(0, h):
                for x in range(0, w):
                    target_idx = target[idx, y, x].item()
                    output_idx = winners[idx, y, x].item()
                    f.write('%d, %d, target, %d, %d, ' % (target_idx, output_idx, x, y))
                    f.write(', '.join(map(str, target_ndvi[idx, :, y, x].data.tolist())))
                    f.write("\n")
                    f.write('%d, %d, predic, %d, %d, ' % (target_idx, output_idx, x, y))
                    f.write(', '.join(map(str, out_ndvi[idx, :, y, x].data.tolist())))
                    f.write("\n")
                    f.write('%d, %d, cls_ai, output, %d, %d, ' % (target_idx, output_idx, x, y))
                    cai = class_activations(out_ndvi, weight, target_idx, idx, y, x)
                    f.write(', '.join(map(str, cai.data.tolist())))
                    f.write("\n")
            # f.write("\n")
            
def get_all_signatures(inp, target, num_cls, b4_index, b8_index):
    """
    expected input having shape  (c, t, h, w) and target (h, w)
        c = number of channels for each sentinel-2 image
        t = number of images in the time series
        hxw = image size
    """
    c, t, h, w = inp.shape
    output_ndvi = np.zeros((t, h, w), dtype=np.float)

    # xin = torch.linspace(1, t, t)

    for cls_index_ in range(0, num_cls):
        pts = (target == cls_index_).numpy()
        all_ndvi_x_cls = []
        for row, yr in enumerate(pts):
            for col, xc in enumerate(yr):
                if xc:  # is True
                    # if target[batch_index_, row, col].item() != cls_index_:
                    #     print("error")
                    b8 = inp[b8_index, :, row, col]
                    b4 = inp[b4_index, :, row, col]
                    ndvi = (b8 - b4) / (b8 + b4)
                    ndvi = np.nan_to_num(ndvi.numpy())
                    # if np.isnan(ndvi).any():
                    #     print("NAN in ndvi!")
                    all_ndvi_x_cls.append(ndvi)
        mean_ndvi = np.zeros((t,), dtype=float)
        if len(all_ndvi_x_cls) > 1:
            mean_ndvi = np.mean(all_ndvi_x_cls, axis=0)
        if len(all_ndvi_x_cls) == 1:
            mean_ndvi = all_ndvi_x_cls[0]
        mmax_ndvi = __max_filter1d_valid(mean_ndvi, 5)  # moving max x class

        # print("batch", batch_index_, ", cls", cls_index_, ", ndvi", mmax_ndvi)
        # plt.plot(xin, mmax_ndvi)

        output_ndvi[:, pts] = mmax_ndvi.reshape(t, 1)
    # plt.show()
    return torch.from_numpy(output_ndvi).float()

# %%
def adjust_classes_weights(cur_epoch, curr_iter, num_iter_x_epoch, tot_epochs, start_w, descending=True):
    current_iter = curr_iter + cur_epoch * num_iter_x_epoch
    max_iter = tot_epochs * num_iter_x_epoch
    # a = 0.75 - current_iter / (2 * max_iter)  # from 0.75 to 0.25
    a = 1 - current_iter / (max_iter)  # from 0 to 1
    if not descending:
        a = 1 - a    # from 0.25 to 0.75

    w = a * start_w + 1 - a
    return w
def adjust_classes_weights(cur_epoch, curr_iter, num_iter_x_epoch, tot_epochs, start_w, descending=True):
    current_iter = curr_iter + cur_epoch * num_iter_x_epoch
    max_iter = tot_epochs * num_iter_x_epoch
    # a = 0.75 - current_iter / (2 * max_iter)  # from 0.75 to 0.25
    a = 1 - current_iter / (max_iter)  # from 0 to 1
    if not descending:
        a = 1 - a    # from 0.25 to 0.75

    w = a * start_w + 1 - a
    return w


def adjust_learning_rate(cur_epoch, curr_iter, num_iter_x_epoch, tot_epochs, start_lr, lr_decay='cos'):
    current_iter = curr_iter + cur_epoch * num_iter_x_epoch
    max_iter = tot_epochs * num_iter_x_epoch

    if lr_decay == 'cos':
        lr = start_lr * (1 + cos(pi * (current_iter) / (max_iter))) / 2
    # elif lr_decay == 'step':
    #     lr = start_lr * (0.1 ** (cur_epoch // args.step))
    elif lr_decay == 'linear':
        lr = start_lr * (1 - (current_iter) / (max_iter))
    # elif lr_decay == 'schedule':
    #     count = sum([1 for s in args.schedule if s <= cur_epoch])
    #     lr = start_lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(lr_decay))

    return lr
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _remove_index(np_matrix, ignore_index=-100):
    if ignore_index == -100:
        return np_matrix
    else:
        m = np.delete(np_matrix, ignore_index, 0) # axes=0 -> delete row index
        m = np.delete(m, ignore_index, 1) # axes=1 -> delete col index
        return m


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, labels, threshold=None, ignore_channels=None):
    """Computes the IoU and mean IoU.
    The mean computation ignores NaN elements of the IoU array.
    Returns:
        Tuple: (IoU, mIoU). The first output is the per class IoU,
        for K classes it's numpy.ndarray with K elements. The second output,
        is the mean IoU.
    """
    # Dimensions check
    assert pr.size(0) == gt.size(0), \
        'number of targets and predicted outputs do not match'
    assert pr.dim() == gt.dim(), \
        "predictions and targets must be of dimension (N, H, W)"

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # sklearn.metrics
    pr1d = pr.view(-1).cpu().detach().numpy()
    gt1d = gt.view(-1).cpu().detach().numpy()
    score = jaccard_score(gt1d, pr1d, labels=labels, average=None)

    return score
    # conf_metric = ConfusionMatrix(num_classes, normalized)
    # conf_metric.add(pr.view(-1), gt.view(-1))
    # conf_matrix = conf_metric.value()
    # if ignore_channels is not None:
    #     conf_matrix[:, ignore_channels] = 0
    #     conf_matrix[ignore_channels, :] = 0
    # true_positive = np.diag(conf_matrix)
    # false_positive = np.sum(conf_matrix, 0) - true_positive
    # false_negative = np.sum(conf_matrix, 1) - true_positive
    #
    # # Just in case we get a division by 0, ignore/hide the error
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     iou = true_positive / (true_positive + false_positive + false_negative)
    #
    # return iou, np.nanmean(iou)


def f_score(pr, gt, labels, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        threshold: threshold for outputs binarization
    Returns:
        numpy array float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # sklearn.metrics
    pr1d = pr.view(-1).cpu().detach().numpy()
    gt1d = gt.view(-1).cpu().detach().numpy()
    score = f1_score(gt1d, pr1d, labels=labels, average=None)

    # tp = torch.sum(gt * pr)
    # fp = torch.sum(pr) - tp
    # fn = torch.sum(gt) - tp
    #
    # score = ((1 + beta ** 2) * tp + eps) \
    #         / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def accuracy(pr, gt, labels, threshold=None, ignore_channels=None, ignore_index=-100):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # sklearn.metrics
    pr1d = pr.view(-1).cpu().detach().numpy()
    gt1d = gt.view(-1).cpu().detach().numpy()

    matrix = confusion_matrix(gt1d, pr1d, labels=labels)
    matrix = _remove_index(matrix, ignore_index)
    diagonal = matrix.diagonal()
    score_per_class = np.nan_to_num(diagonal / matrix.sum(axis=1))

    score = diagonal.sum() / matrix.sum()

    # corrects = (gt == pr).float()
    # score = corrects.sum() / float(corrects.numel())

    return score_per_class, score


def precision(pr, gt, labels, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        numpy array: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # sklearn.metrics
    pr1d = pr.view(-1).cpu().detach().numpy()
    gt1d = gt.view(-1).cpu().detach().numpy()
    score = precision_score(gt1d, pr1d, labels=labels, average=None)

    return score


def recall(pr, gt, labels, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        numpy array float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # sklearn.metrics
    pr1d = pr.view(-1).cpu().detach().numpy()
    gt1d = gt.view(-1).cpu().detach().numpy()
    score = recall_score(gt1d, pr1d, labels=labels, average=None)  # per class measure

    return score

# %%
import numpy

# %%
def compute_train_weights(train_loader):
    beta = 0.9
    samples_per_cls = torch.zeros(n_classes)
    for batch_idx, data in enumerate(train_loader):
        inputs, targets, _, _, _ = data
        for cls in range(n_classes):
            samples_per_cls[cls] += torch.sum(targets == cls)
    max_occ = torch.max(samples_per_cls)
    weights = torch.FloatTensor(max_occ / samples_per_cls)
    # max_occ = torch.max(weights)
    # weights = torch.FloatTensor(weights / max_occ)
    if torch.cuda.is_available():
        weights = weights.cuda()

    return weights

def train_epoch(dataloader, network, optimizer, loss, ep, loss_cls, cls_weights=None):
    num_processed_samples = 0
    num_train_samples = len(dataloader.dataset)
    labels = list(range(numclasses))

    conf_mat_metrics = ConfusionMatrix(labels, ignore_class=ignore_index)
    num_cls = 18
    batch_size = 18
    labels = list(range(num_cls))
    conf_mat = ConfusionMatrix(labels, ignore_class=0)

    for e in range(5):
        target = torch.randint(num_cls, (batch_size,))
        output = torch.rand(batch_size, num_cls)
        pred = torch.argmax(output, dim=1)
        metrics_v, metrics_s = conf_mat(pred, target)
        for k, v in metrics_v.items():
            print(k, v)
        for k, v in metrics_s.items():
            print(k, v)
        print()
    loss_measure = AverageValueMeter()
    am = AverageValueMeter()
    am.add(numpy.array([1, 0.5, 0.2, 0.1, 0.8]))
    mean, std = am.value()
    print(mean, std, am.sum)

    am.add(numpy.array([0.5, 0.5, 0.5, 0.5, 0.5]))
    mean, std = am.value()
    print(mean, std, am.sum)
    var_learning_rate = learning_rate
    for iteration, data in enumerate(dataloader):
        # with torch.no_grad():
        if optimizer == 'sgd':
            var_learning_rate = adjust_learning_rate(ep, iteration,  len(dataloader),
                                      n_epochs, learning_rate, lr_decay='cos')
            for param_group in optimizer.param_groups:
                param_group['lr'] = var_learning_rate
        if cls_weights is not None:
            loss_cls.weight = adjust_classes_weights(ep, iteration, len(dataloader),
                                      n_epochs, cls_weights, descending=False)

        optimizer.zero_grad()

        input, target, target_ndvi, _, _ = data
        num_processed_samples += len(input)
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            target_ndvi = target_ndvi.cuda()

        if loss == 'batch':
            output = network.forward(input)
            samples_per_cls = torch.stack([(target == x_u).sum() for x_u in range(n_classes)])
            l = loss(output, target, samples_per_cls)
        elif loss == 'ndvi':
            out_ndvi, output = network.forward(input)
            samples_per_cls = None
            if loss_weights:
                samples_per_cls = torch.stack([(target == x_u).sum() for x_u in range(n_classes)])
            l = loss(out_ndvi, output, target, target_ndvi, samples_per_cls, weight_mse=1.0)
        else:
            output = network.forward(input)
            l = loss(output, target)

        l.backward()
        optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(output, dim=1).squeeze(1)

            metrics_v, metrics_scalar = conf_mat_metrics(pred, target)
            str_metrics = ''.join(['%s| %f | ' % (key, value) for (key, value) in metrics_scalar.items()])
            loss_measure.add(l.item())
            str_metrics += 'loss| %f | ' % loss_measure.mean

            train_info = 'Train on | {} | Epoch| {} | [{}/{} ({:.0f}%)] | lr| {:.5f} | {}    '.format(
                dataloader.dataset.name, ep, num_processed_samples, num_train_samples,
                100. * (iteration + 1) / len(dataloader), var_learning_rate, str_metrics)
            sys.stdout.write('\r' + train_info)

    print()
    with open(os.path.join(result_path, result_train), 'a+') as f:
        f.write(train_info + '\n')


def test_epoch(dataloader, network, loss):
    num_processed_samples = 0
    num_test_samples = len(dataloader.dataset)
    labels = list(range(numclasses))
    conf_mat_metrics = ConfusionMatrix(labels, ignore_class=ignore_index)
    num_cls = 18
    batch_size = 18
    labels = list(range(num_cls))
    conf_mat = ConfusionMatrix(labels, ignore_class=0)

    for e in range(5):
        target = torch.randint(num_cls, (batch_size,))
        output = torch.rand(batch_size, num_cls)
        pred = torch.argmax(output, dim=1)
        metrics_v, metrics_s = conf_mat(pred, target)
        for k, v in metrics_v.items():
            print(k, v)
        for k, v in metrics_s.items():
            print(k, v)
        print()
    loss_measure = AverageValueMeter()
    am = AverageValueMeter()
    am.add(numpy.array([1, 0.5, 0.2, 0.1, 0.8]))
    mean, std = am.value()
    print(mean, std, am.sum)

    am.add(numpy.array([0.5, 0.5, 0.5, 0.5, 0.5]))
    mean, std = am.value()
    print(mean, std, am.sum)

    with torch.no_grad():
        for iteration, data in enumerate(dataloader):

            input, target, target_ndvi, _, _ = data
            num_processed_samples += len(input)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                target_ndvi = target_ndvi.cuda()

            if loss == 'batch':
                output = network.forward(input)
                samples_per_cls = torch.stack([(target == x_u).sum() for x_u in range(n_classes)])
                l = loss(output, target, samples_per_cls)
            elif loss == 'ndvi':
                out_ndvi, output = network.forward(input)
                samples_per_cls = None
                if loss_weights:
                    samples_per_cls = torch.stack([(target == x_u).sum() for x_u in range(n_classes)])
                l = loss(out_ndvi, output, target, target_ndvi, samples_per_cls)
            else:
                output = network.forward(input)
                l = loss(output, target)

            pred = torch.argmax(output, dim=1).squeeze(1)
            metrics_v, metrics_scalar = conf_mat_metrics(pred, target)
            str_metrics = ''.join(['%s| %f | ' % (key, value) for (key, value) in metrics_scalar.items()])
            loss_measure.add(l.item())
            str_metrics += 'loss| %f | ' % loss_measure.mean
            test_info = 'Test on | {} | Epoch | {} | [{}/{} ({:.0f}%)] | {}  '.format(
                dataloader.dataset.name, epoch, num_processed_samples,
                num_test_samples, 100. * (iteration + 1) / len(dataloader),
                str_metrics)
            sys.stdout.write('\r' + test_info)

        is_best = metrics_scalar['OA'] > best_test_acc
        best = '  **best result' if is_best else '         '
        test_info += best

        sys.stdout.write('\r' + test_info + '\n')
        with open(os.path.join(result_path, result_validation), 'a+') as f:
            f.write(test_info + '\n')

        if is_best:
            cls_names = numpy.array(traindataset.classes)[conf_mat_metrics.get_labels()]
            with open(os.path.join(result_path, "per_class_metricsDeepLab.txt"), 'a+') as f:
                f.write('classes:\n' + numpy.array2string(cls_names) + '\n')
                for k, v in metrics_v.items():
                    f.write(k + '\n')
                    if len(v.shape) == 1:
                        for ki, vi in zip(cls_names, v):
                            f.write("%.2f" % vi + '\t' + ki + '\n')
                    elif len(v.shape) == 2:  # confusion matrix
                        num_gt = numpy.sum(v, axis=1)
                        f.write('\n'.join(
                            [''.join(['{:10}'.format(item) for item in row]) + '  ' + lab + '(%d)' % tot
                             for row, lab, tot in zip(v, cls_names, num_gt)]))
                        f.write('\n')

        return metrics_scalar['OA']  # test_acc


def test_only(dataloader, network, loss, epoch, set_name):
    num_processed_samples = 0
    num_test_samples = len(dataloader.dataset)
    labels = list(range(numclasses))
    conf_mat_metrics = ConfusionMatrix(labels, ignore_class=ignore_index)
    num_cls = 18
    batch_size = 18
    labels = list(range(num_cls))
    conf_mat = ConfusionMatrix(labels, ignore_class=0)

    for e in range(5):
        target = torch.randint(num_cls, (batch_size,))
        output = torch.rand(batch_size, num_cls)
        pred = torch.argmax(output, dim=1)
        metrics_v, metrics_s = conf_mat(pred, target)
        for k, v in metrics_v.items():
            print(k, v)
        for k, v in metrics_s.items():
            print(k, v)
        print()
    loss_measure = AverageValueMeter()
    am = AverageValueMeter()
    am.add(numpy.array([1, 0.5, 0.2, 0.1, 0.8]))
    mean, std = am.value()
    print(mean, std, am.sum)

    am.add(numpy.array([0.5, 0.5, 0.5, 0.5, 0.5]))
    mean, std = am.value()
    print(mean, std, am.sum)

    with torch.no_grad():
        for iteration, data in enumerate(dataloader):

            input, target, target_ndvi, dates, patch_id = data

            num_processed_samples += len(input)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                target_ndvi = target_ndvi.cuda()

            if loss == 'batch':
                output = network.forward(input)
                samples_per_cls = torch.stack([(target == x_u).sum() for x_u in range(n_classes)])
                l = loss(output, target, samples_per_cls)
            elif loss == 'ndvi':
                out_ndvi, output = network.forward(input)
                samples_per_cls = None
                if loss_weights:
                    samples_per_cls = torch.stack([(target == x_u).sum() for x_u in range(n_classes)])
                l = loss(out_ndvi, output, target, target_ndvi, samples_per_cls)
                if iteration % 100 == 0:
                    write_signatures(model.module, target, output, target_ndvi, out_ndvi, dates, patch_id, set_name)
            else:
                output = network.forward(input)
                l = loss(output, target)

            pred = torch.argmax(output, dim=1).squeeze(1)
            metrics_v, metrics_scalar = conf_mat_metrics(pred, target)
            str_metrics = ''.join(['%s| %f | ' % (key, value) for (key, value) in metrics_scalar.items()])
            loss_measure.add(l.item())
            str_metrics += 'loss| %f | ' % loss_measure.mean
            test_info = 'Test on | {} | Epoch | {} | [{}/{} ({:.0f}%)] | {}  '.format(
                dataloader.dataset.name, epoch, num_processed_samples,
                num_test_samples, 100. * (iteration + 1) / len(dataloader),
                str_metrics)
            if ((100. * (iteration + 1))%100 == 0):
                sys.stdout.write('\r' + test_info)
            

        cls_names = numpy.array(traindataset.classes)[conf_mat_metrics.get_labels()]
        if ((100. * (iteration + 1))%100 == 0):
            sys.stdout.write('\r' + test_info)
        with open(os.path.join(result_path, set_name + "_per_class_metricsDeepLab.txt"), 'w') as f:
            f.write('classes:\n' + numpy.array2string(cls_names) + '\n')
            sys.stdout.write('classes:\n' + numpy.array2string(cls_names) + '\n')
            for k, v in metrics_v.items():
                sys.stdout.write('\n' + k + '\n')
                f.write('\n' + k + '\n')
                if len(v.shape) == 1:
                    for ki, vi in zip(cls_names, v):
                        sys.stdout.write("%.2f" % vi + '\t' + ki + '\n')
                        f.write("%.2f" % vi + '\t' + ki + '\n')
                elif len(v.shape) == 2:  # confusion matrix
                    num_gt = numpy.sum(v, axis=1)
                    sys.stdout.write('\n'.join(
                        [''.join(['{:10}'.format(item) for item in row]) + '  ' + lab + '(%d)' % tot
                         for row, lab, tot in zip(v, cls_names, num_gt)]))
                    f.write('\n'.join(
                        [''.join(['{:10}'.format(item) for item in row]) + '  ' + lab + '(%d)' % tot
                         for row, lab, tot in zip(v, cls_names, num_gt)]))
                    sys.stdout.write('\n')
                    f.write('\n')

        if loss == 'ndvi':
            print("\nClass Activation Interval saved in:", result_path)

# %%
class ASPP(nn.Module):
    def __init__(self, num_classes):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv3d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm3d(256)

        self.conv_3x3_1 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm3d(256)

        self.conv_3x3_2 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm3d(256)

        self.conv_3x3_3 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm3d(256)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv_1x1_2 = nn.Conv3d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm3d(256)

        self.conv_1x1_3 = nn.Conv3d(1280, 256, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm3d(256)

        self.conv_1x1_4 = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]
        feature_map_c = feature_map.size()[4]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w, feature_map_c), mode='trilinear', align_corners=True)

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        out = self.conv_1x1_4(out)


        return out

class ASPP_Bottleneck(nn.Module):
    def __init__(self, num_classes):
        super(ASPP_Bottleneck, self).__init__()

        self.conv_1x1_1 = nn.Conv3d(4*512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm3d(256)

        self.conv_3x3_1 = nn.Conv3d(4*512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm3d(256)

        self.conv_3x3_2 = nn.Conv3d(4*512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm3d(256)

        self.conv_3x3_3 = nn.Conv3d(4*512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm3d(256)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv_1x1_2 = nn.Conv3d(4*512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm3d(256)

        self.conv_1x1_3 = nn.Conv3d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm3d(256)

        self.conv_1x1_4 = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]
        feature_map_c = feature_map.size()[4]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w, feature_map_c), mode='trilinear', align_corners=True)
        
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        out = self.conv_1x1_4(out)

        return out


class DeepLabV3_3D(nn.Module):
    def __init__(self, num_classes, input_channels, resnet, last_activation = None):
        super(DeepLabV3_3D, self).__init__()
        self.num_classes = num_classes
        self.last_activation = last_activation

        if resnet.lower() == 'resnet18_os16':
            self.resnet = ResNet18_OS16(input_channels)
        
        elif resnet.lower() == 'resnet34_os16':
            self.resnet = ResNet34_OS16(input_channels)
        
        elif resnet.lower() == 'resnet50_os16':
            self.resnet = ResNet50_OS16(input_channels)
        
        elif resnet.lower() == 'resnet101_os16':
            self.resnet = ResNet101_OS16(input_channels)
        
        elif resnet.lower() == 'resnet152_os16':
            self.resnet = ResNet152_OS16(input_channels)
        
        elif resnet.lower() == 'resnet18_os8':
            self.resnet = ResNet18_OS8(input_channels)
        
        elif resnet.lower() == 'resnet34_os8':
            self.resnet = ResNet34_OS8(input_channels)

        if resnet.lower() in ['resnet50_os16', 'resnet101_os16', 'resnet152_os16']:
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)
        else:
            self.aspp = ASPP(num_classes=self.num_classes)

        num_classes = 18
        self.final_conv = torch.nn.Conv3d(30, 1, kernel_size=1, stride=1, padding=0, bias=True)
        

    def forward(self, x):

        h = x.size()[2]
        w = x.size()[3]
        c = x.size()[4]

        feature_map = self.resnet(x)

        output = self.aspp(feature_map)

        output = F.interpolate(output, size=(h, w, c), mode='trilinear', align_corners=True)
        
        if self.last_activation.lower() == 'sigmoid':
            output = nn.Sigmoid()(output)
        
        elif self.last_activation.lower() == 'softmax':
            output = nn.Softmax()(output)

        
        output = torch.permute(output, (0,2,1,3,4))
        print(output.shape)
        output = self.final_conv(output)
        output = torch.squeeze(output)
        
        
        return output

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, input_channels, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(input_channels, **kwargs):
    model = ResNet(input_channels, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(input_channels, **kwargs):
    model = ResNet(input_channels, BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(input_channels, **kwargs):
    model = ResNet(input_channels, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(input_channels, **kwargs):
    model = ResNet(input_channels, Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(input_channels, **kwargs):
    model = ResNet(input_channels, Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


# -------------------------------------- Resnet for Deeplab --------------------------------------
def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1)

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)

    return layer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion*channels
        
        if type(dilation) != type(1):
            dilation = 1
            
        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm3d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + self.downsample(x)

        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        self.conv3 = nn.Conv3d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm3d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + self.downsample(x)

        out = F.relu(out)

        return out

class ResNet_Bottleneck_OS16(nn.Module):
    def __init__(self, num_layers, input_channels):
        super(ResNet_Bottleneck_OS16, self).__init__()

        if num_layers == 50:
            resnet = resnet50(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        elif num_layers == 101:
            resnet = resnet101(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        elif num_layers == 152:
            resnet = resnet152(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        else:
            raise Exception("num_layers must be in {50, 101, 152}!")

        self.layer5 = make_layer(Bottleneck, in_channels=4*256, channels=512, num_blocks=3, stride=1, dilation=2)

    def forward(self, x):
        c4 = self.resnet(x)

        output = self.layer5(c4)

        return output

class ResNet_BasicBlock_OS16(nn.Module):
    def __init__(self, num_layers, input_channels):
        super(ResNet_BasicBlock_OS16, self).__init__()

        if num_layers == 18:
            resnet = resnet18(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 2

        elif num_layers == 34:
            resnet = resnet34(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")
    
        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks, stride=1, dilation=2)

    def forward(self, x):
        c4 = self.resnet(x)

        output = self.layer5(c4)

        return output

class ResNet_BasicBlock_OS8(nn.Module):
    def __init__(self, num_layers, input_channels):
        super(ResNet_BasicBlock_OS8, self).__init__()

        if num_layers == 18:
            resnet = resnet18(input_channels)
            
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 2
            num_blocks_layer_5 = 2

        elif num_layers == 34:
            resnet = resnet34(input_channels)
            
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 6
            num_blocks_layer_5 = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1, dilation=2)

        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1, dilation=4)

        

    def forward(self, x):
        c3 = self.resnet(x)

        output = self.layer4(c3)
        output = self.layer5(output)
        

        return output

def ResNet18_OS16(input_channels):
    return ResNet_BasicBlock_OS16(num_layers=18, input_channels=input_channels)

def ResNet50_OS16(input_channels):
    return ResNet_Bottleneck_OS16(num_layers=50, input_channels=input_channels)

def ResNet101_OS16(input_channels):
    return ResNet_Bottleneck_OS16(num_layers=101, input_channels=input_channels)

def ResNet152_OS16(input_channels):
    return ResNet_Bottleneck_OS16(num_layers=152, input_channels=input_channels)

def ResNet34_OS16(input_channels):
    return ResNet_BasicBlock_OS16(num_layers=34, input_channels=input_channels)

def ResNet18_OS8(input_channels):
    return ResNet_BasicBlock_OS8(num_layers=18, input_channels=input_channels)

def ResNet34_OS8(input_channels):
    return ResNet_BasicBlock_OS8(num_layers=34, input_channels=input_channels)

# %%
input_channels = 13          
resnet = 'resnet34_os8'    
last_activation = 'softmax' 

model = DeepLabV3_3D(num_classes = 18, input_channels = input_channels, resnet = resnet, last_activation = last_activation).cuda()

# %%
labels = list(range(numclasses))

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
if ottimizzatore == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
elif ottimizzatore == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

# Define Criterion
weights = None
if not (loss == 'batch' or loss == 'ndvi') and loss_weights:
    print('Computing weights per classes...')
    weights = compute_train_weights(traindataloader)
    weights = torch.sqrt(weights)
    print("weights per classes:", weights)

loss_cls = SegmentationLosses(weight=weights, cuda=True, ignore_index=ignore_index)
loss = loss_cls.build_loss(mode='ce')




start_epoch = 0
if resume_path:  # Empty strings are considered false
    print('trying to resume previous saved model...')
    state = resume(resume_path, model=model, optimizer=optimizer)

    if "epoch" in state.keys():
        start_epoch = state["epoch"]
        best_test_acc = state['best_test_acc']


for epoch in range(start_epoch, n_epochs):
    train_epoch(traindataloader, model, optimizer, loss, epoch, loss_cls, cls_weights=None)
    val_acc = test_epoch(validationdataloader, model, loss)

    is_best = val_acc > best_test_acc
    if is_best:
        epochs_best_acc = epoch
        best_test_acc = val_acc
        if result_path:
            checkpoint_name = os.path.join(result_path, "best_model.pth")
            save(checkpoint_name, model, optimizer,
                epoch=epoch, best_test_acc=best_test_acc)

# %%



