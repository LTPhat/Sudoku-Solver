from PIL import Image, ImageFont, ImageDraw
import random
import glob
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import torch
import time
import copy
from create_dataset.digital_mnist_digits import PrintedMNIST
from get_model import get_model

# Define parameters
batch_size = 64
net = get_model("resnet50")
n_epochs = 10
device = "cuda" if torch.cuda.is_available() == True else "cpu"

#Define optimizer
learning_rate = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


def load_dataset(batch_size):
    """
    Load dataset using Pytorch Dataloader
    """
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # AddGaussianNoise(0, 1.0),
        # AddSPNoise(0.1),

    ])
    val_transforms = transforms.Compose([transforms.ToTensor()])

    train_set = PrintedMNIST(50000, 42, train_transform)
    val_set = PrintedMNIST(5000, 33, val_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_track = []
    train_acc_track = []
    val_loss_track = []
    val_acc_track = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training loop
        train_loss, train_correct = 0, 0  
        model.train()  
        for batch in train_loader:
            images, labels = batch[0].to(device), batch[1].to(device)  # load the batch to the available device (cpu/gpu)
            outputs = model(images)  
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1).cpu().numpy()
            optimizer.zero_grad()   
            loss.backward()  
            optimizer.step()  
            np_labels_train = labels.cpu().numpy()
            train_loss += loss.item() * batch_size  
            train_correct += np.sum(preds == np_labels_train)
        train_loss_avg = train_loss/len(train_loader.sampler)
        train_acc_avg = train_correct/len(train_loader.sampler)
        print('Train Loss: ',train_loss_avg)  
        print('Train Accuracy: ', train_acc_avg)
        train_loss_track.append(train_loss_avg)
        train_acc_track.append(train_acc_avg)

        #Validation loop
        model.eval()  
        with torch.no_grad(): 
            valid_loss, valid_correct = 0, 0 

            for batch in val_loader:
                images, labels = batch[0].to(device), batch[1].to(device)  # load the batch to the available device
                outputs = model(images)  
                loss = criterion(outputs, labels)  
                preds = outputs.argmax(dim=1).cpu().numpy()
                np_label_val = labels.cpu().numpy()
                valid_loss += loss.item() * batch_size  
                valid_correct += np.sum(preds == np_label_val)
            if valid_correct > best_acc:
                best_acc = valid_correct
                best_model_wts = copy.deepcopy(model.state_dict())
            valid_loss_avg = valid_loss/len(val_loader.sampler)
            valid_acc_avg = valid_correct/len(val_loader.sampler)
            print('Validation Loss: ', valid_loss_avg)  
            print('Validation Accuracy: ',valid_acc_avg)
            val_loss_track.append(valid_loss_avg)
            val_acc_track.append(valid_acc_avg)
            
    # Return model with best metrics          
    model.load_state_dict(best_model_wts)
    return model, train_loss_track, train_acc_track, val_loss_track, val_acc_track

if __name__ == "__main__":
    train_loader, val_loader = load_dataset(batch_size)
    model, train_loss, train_acc, val_loss, val_acc = train(net, train_loader, val_loader, criterion, optimizer, n_epochs)
