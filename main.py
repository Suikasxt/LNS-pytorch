import argparse
import torch
import numpy as np
from utils import load_data, SampleDataset
from torch.utils.data import DataLoader
from model import BaseModel
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--ns_method', default='sample')
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--gpu', default=0, type=int)
args = parser.parse_args()
device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() and args.gpu != -1 else torch.device('cpu')

def prepareData(ns_method = 'sample'):
    n_feature, n_label, feature, label = load_data('./Amazon670K/train.txt', device)
    train_dataloader = DataLoader(dataset=SampleDataset(n_feature, n_label, feature, label),
                                  batch_size=args.batch_size)
    n_feature, n_label, feature, label = load_data('./Amazon670K/test.txt', device)
    test_dataloader = DataLoader(dataset=SampleDataset(n_feature, n_label, feature, label, mode='test'),
                                 batch_size=args.batch_size)
    
    return n_feature, n_label, train_dataloader, test_dataloader

def test(model, test_dataloader):
    result = []
    for (features, pos_label) in tqdm(test_dataloader):
        logits = model.forward(features)
        for index in range(features.shape[0]):
            predict = torch.argmax(logits[index])
            result.append(predict in pos_label[index])
    print("P @ 1: {}".format(np.mean(result)))
            

def train(model, train_dataloader, test_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epoch):
        loss_list = []
        for (features, pos_label, neg_label) in tqdm(train_dataloader, f"Training at epoch {epoch}"):
            features = features.to(device)
            logits = model.forward(features)
            index = torch.arange(pos_label.shape[0], dtype=torch.int)
            loss = torch.sum((logits[index, pos_label] - 1)**2)
            loss += torch.sum((logits[index.view(-1, 1), neg_label])**2)
            
            loss_list.append(loss.items())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch #{} loss: {}".format(epoch, np.mean(loss_list)))
        test(model, test_dataloader)

if __name__ == "__main__":
    n_feature, n_label, train_dataloader, test_dataloader = prepareData(args.ns_method)
    print('Data prepared!')
    model = BaseModel(n_feature, n_label, 128)
    model.to(device)
    
    if args.mode == 'train':
        train(model, train_dataloader, test_dataloader)
    
