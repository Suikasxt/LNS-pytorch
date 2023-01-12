import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

class SampleDataset(Dataset):
    def __init__(self, n_feature, n_label, features, labels, mode='train', neg_num=10):
        self.n_feature = n_feature
        self.n_label = n_label
        self.features = features
        self.labels = labels
        self.mode = mode
        self.neg_num = neg_num

    def __getitem__(self, index):
        if self.mode == 'train':
            neg_item = torch.randint(0, self.n_label, (self.neg_num,))
            pos_item = np.random.choice(self.labels[index], 1)
            return self.features[index], torch.IntTensor(pos_item), neg_item
        else:
            return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)


def load_data(path, device):
    data_label = []
    data_feature = []
    data_feature_tensor = []
    with open(path, 'r') as f:
        n_point, n_feature, n_label = [int(x) for x in f.readline().split()]
        
        if False: # reload
            for line in tqdm(f.readlines()):
                sp = line.split(' ')
                labels = [int(x) for x in sp[0].split(',')]
                k = [int(f.split(':')[0]) for f in sp[1:]]
                v = [float(f.split(':')[1]) for f in sp[1:]]
                data_label.append(labels)
                data_feature.append([k, v])
            with open(path+'_label.pkl', 'wb') as f:
                pickle.dump(data_label, f)
            with open(path+'_feature.pkl', 'wb') as f:
                pickle.dump(data_feature, f)
        else:
            with open(path+'_label.pkl', 'rb') as f:
                data_label = pickle.load(f)
            with open(path+'_feature.pkl', 'rb') as f:
                data_feature = pickle.load(f)
        
    for (k, v) in tqdm(data_feature):
        data_feature_tensor.append(torch.sparse_coo_tensor([k], v, (n_feature,)))
        
    return n_feature, n_label, data_feature_tensor, data_label