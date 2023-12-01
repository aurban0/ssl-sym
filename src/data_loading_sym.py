from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

# For MNIST with partial rotations
def PartialMNIST_AE_Dataloader(config, train=True, test=False, shuffle=False, custom_batchsize=0, theta=False,
                               pseudolabels=None, return_index=False, no_val_split=False):
    print("Loading for train:",train,", and for test:",test)
    class MNISTRotDataset(Dataset):
        def __init__(self, pseudolabels):
            if train:
                self.data = pd.read_pickle(
                    config.customdata_train_path).to_numpy()
            if test:
                self.data = pd.read_pickle(
                    config.customdata_test_path).to_numpy()
            self.x = self.data[:, :-1].reshape(len(self.data), 28, 28)
            self.y = self.data[:, -1]
            self.num_samples = len(self.x)
            if theta:
                self.pseudolabels = pseudolabels

        def __len__(self):
            return self.num_samples

        def __getitem__(self, item):
            x = torch.from_numpy(self.x[item]).float()
            y = torch.from_numpy(np.array(self.y[item])).float()
            x = x.unsqueeze(0)
            if return_index:
                return x, y, item
            if theta:
                pseudolabel = self.pseudolabels[item].float()
                return x, y, pseudolabel
            else:
                return x, y

    dataset = MNISTRotDataset(pseudolabels=pseudolabels)
    # Train and validation datasets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size_value = int(custom_batchsize) if custom_batchsize else config.dataloader_batch_sz
    if no_val_split:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size_value,
                                                   shuffle=shuffle,
                                                   num_workers=0,
                                                   drop_last=False)
        return [dataloader]

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size_value,
                                               shuffle=shuffle,
                                               num_workers=0,
                                               drop_last=False)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size_value,
                                             shuffle=False,
                                             num_workers=0,
                                             drop_last=False)

    return [train_loader, val_loader]

# For RotMNIST or MNIST benchmarks (.amat)
def RotMNIST_AE_Dataloader(config, train=True, test=False, custom_batchsize=0, shuffle=False,
                           theta=False, pseudolabels=None,
                           return_index=False, no_val_split=False):
    print("Loading for train:",train,", and for test:",test)
    class MNISTRotationDataset(Dataset):
        def __init__(self, train=train, test=test, pseudolabels=pseudolabels):
            self.train = train
            self.test = test
            if self.train:
                self.data = np.loadtxt(config.customdata_train_path)
            elif self.test:
                self.data = np.loadtxt(config.customdata_test_path)
            self.num_samples = len(self.data)
            self.x = self.data[:, :-1].reshape(len(self.data), 28, 28)
            self.y = self.data[:, -1]
            if theta:
                self.pseudolabels = pseudolabels

        def __len__(self):
            return self.num_samples

        def __getitem__(self, index):
            x = torch.from_numpy(self.x[index]).float()
            y = torch.from_numpy(np.array(self.y[index])).float()
            x = x.unsqueeze(0)
            if return_index:
                return x, y, index
            if theta:
                pseudolabel = self.pseudolabels[index].float()
                return x, y, pseudolabel
            else:
                return x, y

    dataset = MNISTRotationDataset(train=train, test=test, pseudolabels=pseudolabels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


    batch_size_value = int(custom_batchsize) if custom_batchsize else config.dataloader_batch_sz
    if no_val_split:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size_value,
                                                   shuffle=shuffle,
                                                   num_workers=0,
                                                   drop_last=False)
        return [dataloader]
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size_value,
                                               shuffle=shuffle,
                                               num_workers=0,
                                               drop_last=False)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size_value,
                                             shuffle=False,
                                             num_workers=0,
                                             drop_last=False)
    return [train_loader, val_loader]