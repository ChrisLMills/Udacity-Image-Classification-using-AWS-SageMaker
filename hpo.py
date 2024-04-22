
import numpy as np
import io
import boto3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import argparse

s3client = boto3.client('s3')


def train(model, train_loader, criterion, optimizer, epoch):
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

def test(model, test_loader):
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
        
    
    
def net(num_classes, trainable_layers):
    
    model = resnet50(pretrained=True)
    layer_count = sum(1 for _ in model.children())
    child_counter = 1
    
    for child in model.children():
        if (layer_count - child_counter) >= trainable_layers:
            for param in child.parameters():
                param.requires_grad = False
 
        child_counter += 1
        
    #for param in model.parameters():
    #    param.requires_grad = False       

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, num_classes))
    return model


def unpickle(file_path):
    import pickle
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def images_reshape(data_array):
    images = []
        
    for i in range(20): #range(len(data_array)):
        row = data_array[i,:]
        red = row[:1024]
        green = row[1024:2048]
        blue = row[2048:]
        
        red2d = red.reshape(32,32)
        green2d = green.reshape(32,32)
        blue2d = blue.reshape(32,32)

        image = np.stack((red2d, green2d, blue2d))
        #print(image)
        images.append(image)
        #print("Stack: ", images)
        
    #print(np.array(images))
    return np.array(images)


class MyDataset(Dataset):
    def __init__(self, data_path):
        dictionary = unpickle(data_path)
        data_array = dictionary[b'data']/255
        target_array = dictionary[b'labels'][:20]

        self.data = torch.from_numpy(images_reshape(data_array)).float()
        self.target = torch.from_numpy(np.array(target_array))
        

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        return x, y

    def __len__(self):
        return len(self.data)
    

def create_data_loaders(batch_size):
    
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    
    s3client.download_file('udacity-module3-project', 'data/batch_1', 'train_batch_1')
    s3client.download_file('udacity-module3-project', 'data/test_batch', 'test_batch')
    
    train_dataset = MyDataset('train_batch_1')
    test_dataset = MyDataset('test_batch')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def main(args):
      
    model=net(args.num_classes, args.trainable_layers)
    
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_loader, test_loader = create_data_loaders(args.batch_size)
    
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, loss_criterion, optimizer, epoch)
        test(model, test_loader)
    
    bucket = 'udacity-module3-project'
    file_path = 'model/output_model_file.bin'
    
    buffer = io.BytesIO()
    torch.save(model, buffer)
    s3client.put_object(Bucket=bucket, Key=file_path, Body=buffer.getvalue())

    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_classes", type = int)
    parser.add_argument("--epochs", type = int)
    parser.add_argument("--batch_size", type = int)
    parser.add_argument("--lr", type = float)
    parser.add_argument("--trainable_layers", type = int)
    
    args = parser.parse_args()
    
    main(args)
