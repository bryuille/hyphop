import numpy as np
import scipy.io
import os
import pickle
import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold

import torch
import torch.utils.data
from torchvision import datasets, transforms

from .loader_utils import *
import random

def load_mnist(batch_size,norm_factor=1):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./mnist_data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)
    trainset = list(iter(trainloader))

    testset = datasets.MNIST(root='./mnist_data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True)
    testset = list(iter(testloader))
    return trainset, testset

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y
        for i in range(len(self.y)):
            if self.y[i] == -1 or self.y[i] == False:
                self.y[i] = 0.0
            else:
                self.y[i] = 1.0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        batch_x = torch.tensor(self.x[idx], dtype=torch.float32)
        batch_y = torch.tensor(self.y[idx], dtype=torch.float32)
        return batch_x, batch_y

    def collate(self, batch):
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        pad_batch_x, mask_x = self.padding(x)
        return pad_batch_x, torch.stack(y, dim=0), mask_x

    def padding(self, batch):
        max_bag_len = max([len(xi) for xi in batch])
        feat_dim = batch[0].size(-1)
        batch_x_tensor = torch.zeros((len(batch), max_bag_len, feat_dim))
        mask_x = torch.ones((len(batch), max_bag_len), dtype=torch.bool)

        for i in range(len(batch)):
            bag_size = batch[i].size(0)
            batch_x_tensor[i, :bag_size] = batch[i]
            mask_x[i][:bag_size] = False
        return batch_x_tensor, mask_x

def load_data(args):
    features = []
    labels = []
    dataset = scipy.io.loadmat(f'./datasets/mil_datasets/{args.dataset}_100x100_matlab.mat')
    instance_bag_ids = np.array(dataset['bag_ids'])[0]
    instance_features = np.array(dataset['features'].todense())
    
    if args.multiply:
        instance_features = multiply_features(instance_features)

    instance_labels = np.array(dataset['labels'].todense())[0]
    bag_features = into_dictionary(instance_bag_ids, instance_features)
    bag_labels = into_dictionary(instance_bag_ids, instance_labels)
    
    for i in sorted(bag_features.keys()):
        features.append(np.array(bag_features.pop(i)))
        labels.append(float(max(bag_labels[i])))
    return features, labels

def get_dataset(args, dataset='fox'):
    if args.multiply:
        filepath = './datasets/mil_datasets/{}_dataset.pkl'.format(args.dataset)
    else:
        filepath = './datasets/mil_datasets/{}_original_dataset.pkl'.format(args.dataset)
    
    dataset_obj = Dataset(args, dataset)
    file = open(filepath, 'wb')
    pickle.dump(dataset_obj, file)
    return dataset_obj

class Dataset():
    def __init__(self, args, dataset='fox'):
        self.rs = args.rs
        self.features = []
        self.bag_labels = []
        dataset_mat = scipy.io.loadmat(f'./datasets/mil_datasets/{dataset}_100x100_matlab.mat')
        instance_bag_ids = np.array(dataset_mat['bag_ids'])[0]
        instance_features = np.array(dataset_mat['features'].todense())
        
        if args.multiply:
            instance_features = multiply_features(instance_features)

        instance_labels = np.array(dataset_mat['labels'].todense())[0]
        bag_features = into_dictionary(instance_bag_ids, instance_features)
        bag_labels = into_dictionary(instance_bag_ids, instance_labels)
        
        for i in sorted(bag_features.keys()):
            self.features.append(np.array(bag_features.pop(i)))
            self.bag_labels.append(float(max(bag_labels[i])))
        self.random_shuffle()

    def random_shuffle(self):
        self.features, self.bag_labels = shuffle_dataset(self.features, self.bag_labels, self.rs)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            self.features, self.bag_labels, test_size=0.2, random_state=self.rs)
        self.training_data = x_train
        self.testing_data = x_test
        self.training_labels = y_train
        self.testing_labels = y_test

    def return_training_set(self):
        return DummyDataset(self.training_data, self.training_labels)

    def return_testing_set(self):
        return DummyDataset(self.testing_data, self.testing_labels)

    def return_dataset(self):
        return DummyDataset(self.features, self.bag_labels)