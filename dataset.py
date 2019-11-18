import torch
from torch import Tensor
from torch.utils import data

import pandas as pd
import numpy as np
import datetime as dt

import os
import json


class PixelSetData(data.Dataset):
    def __init__(self, folder, labels, vector_length, sub_classes=None, norm=None,
                 extra_feature=None, jitter=None):
        super(PixelSetData, self).__init__()

        self.folder = folder
        self.data_folder = os.path.join(folder, 'DATA')
        self.meta_folder = os.path.join(folder, 'META')
        self.labels = labels
        self.vector_length = vector_length
        self.norm = norm

        self.extra_feature = extra_feature
        self.jitter = jitter  # (sigma , clip )



        l = [f for f in os.listdir(self.data_folder) if f.endswith('.npy')]
        self.pid = [int(f.split('.')[0]) for f in l]
        self.pid = list(np.sort(self.pid))

        self.pid = list(map(str, self.pid))
        self.len = len(self.pid)

        # Get Labels
        if sub_classes is not None:
            sub_indices = []
            num_classes = len(sub_classes)
            convert = dict((c, i) for i, c in enumerate(sub_classes))

        with open(os.path.join(folder, 'META', 'labels.json'), 'r') as file:
            d = json.loads(file.read())
            self.target = []
            for i, p in enumerate(self.pid):
                t = d[labels][p]
                self.target.append(t)
                if sub_classes is not None:
                    if t in sub_classes:
                        sub_indices.append(i)
                        self.target[-1] = convert[self.target[-1]]
        if sub_classes is not None:
            self.pid = list(np.array(self.pid)[sub_indices])
            self.target = list(np.array(self.target)[sub_indices])
            self.len = len(sub_indices)

        with open(os.path.join(folder, 'META', 'dates.json'), 'r') as file:
            d = json.loads(file.read())
        self.dates = [d[str(i)] for i in range(len(d))]
        self.date_positions = date_positions(self.dates)


        if self.extra_feature is not None:
            with open(os.path.join(self.meta_folder, '{}.json'.format(extra_feature)), 'r') as file:
                self.extra = json.loads(file.read())

            if isinstance(self.extra[list(self.extra.keys())[0]], int):
                for k in self.extra.keys():
                    self.extra[k] = [self.extra[k]]
            df = pd.DataFrame(self.extra).transpose()
            self.extra_m, self.extra_s = np.array(df.mean(axis=0)), np.array(df.std(axis=0))

        self.indices = list(range(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, item):

        item = self.indices[item]

        x0 = np.load(os.path.join(self.folder, 'DATA', '{}.npy'.format(self.pid[item])))
        y = self.target[item]

        if x0.shape[-1] > self.vector_length:
            idx = np.random.choice(list(range(x0.shape[-1])), size=self.vector_length, replace=False)
            x = x0[:, :, idx]
            mask = np.ones(self.vector_length)

        elif x0.shape[-1] < self.vector_length:

            if x0.shape[-1] == 0:
                x = np.zeros((*x0.shape[:2], self.vector_length))
                mask = np.zeros(self.vector_length)
                mask[0] = 1
            else:
                x = np.zeros((*x0.shape[:2], self.vector_length))
                x[:, :, :x0.shape[-1]] = x0
                x[:, :, x0.shape[-1]:] = np.stack([x[:, :, 0] for _ in range(x0.shape[-1], x.shape[-1])], axis=-1)
                mask = np.array(
                    [1 for _ in range(x0.shape[-1])] + [0 for _ in range(x0.shape[-1], self.vector_length)])
        else:
            x = x0
            mask = np.ones(self.vector_length)

        if self.norm is not None:
            m, s = self.norm
            m = np.array(m)
            s = np.array(s)

            if len(m.shape) == 0:
                x = (x - m) / s
            elif len(m.shape) == 1:  # Normalise channel-wise
                x = (x.swapaxes(1, 2) - m) / s
                x = x.swapaxes(1, 2)  # Normalise channel-wise for each date
            elif len(m.shape) == 2:
                x = np.rollaxis(x, 2)  # TxCxS -> SxTxC
                x = (x - m) / s
                x = np.swapaxes((np.rollaxis(x, 1)), 1, 2)
        x = x.astype('float')

        if self.jitter is not None:
            sigma, clip = self.jitter
            x = x + np.clip(sigma * np.random.randn(*x.shape), -1 * clip, clip)

        mask = np.stack([mask for _ in range(x.shape[0])], axis = 0) #Add temporal dimension to mask
        data = (Tensor(x), Tensor(mask))

        if self.extra_feature is not None:
            ef = (self.extra[str(self.pid[item])] - self.extra_m) / self.extra_s
            ef = torch.from_numpy(ef).float()
            if self.unitemp is None:
                ef = torch.stack([ef for _ in range(data[0].shape[0])], dim=0)
            data = (data, ef)


        return data, torch.from_numpy(np.array(y, dtype=int))




def parse(date):
    d = str(date)
    return int(d[:4]), int(d[4:6]), int(d[6:])


def interval_days(date1, date2):
    return abs((dt.datetime(*parse(date1)) - dt.datetime(*parse(date2))).days)

def date_positions(dates):
    pos = []
    for d in dates:
        pos.append(interval_days(d, dates[0]))
    return pos
