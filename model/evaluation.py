from PIL import Image, ImageFont, ImageDraw
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from train_classifier import model, device, train_loader, val_loader, train_acc, train_loss, val_loss, val_acc

# Just visualize model results

def visualize_sample():
    """
    Visualize sample in dataloader
    """
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")



def plot_metrics(train_loss, train_acc, val_loss, val_acc):
    fig, ag = plt.subplots(1,2,figsize = (15,6))
    ag[0].plot(train_loss,label = 'train')
    ag[0].plot(val_loss,label = 'val')
    ag[0].legend()
    ag[0].set_title('Loss versus epochs')

    ag[1].plot(train_acc,label='train')
    ag[1].plot(val_acc,label='test')
    ag[1].legend()
    ag[1].set_title('Accuracy versus epochs')
    plt.show()


def predict_batch(model, data_loader):
    """
    Get prediction on one random batch
    """

    batch_id = np.random.randint(0, len(data_loader))
    for index, batch in enumerate(data_loader):
        if index == batch_id:
            inputs, labels = batch[0], batch[1]
    model = model.to(device)
    inputs = inputs.to(device)
    outputs = model(inputs)
    preds = outputs.argmax(dim=1)
    preds=preds.cpu().numpy()
    labels=labels.numpy()
    return inputs, preds, labels


if __name__ == "__main__":
    visualize_sample()
    plot_metrics(train_loss, train_acc, val_loss, val_acc)
    inputs, preds, labels = predict_batch(model, val_loader)
    print(preds)
    print(labels)
    print("Accuracy on random batch: {}/{}".format(np.sum(preds==labels), len(preds)))