import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        # (TODO) Design your CNN, it can only be less than 3 convolution layers
        super(CNN, self).__init__()
        n1 = 64
        n2 = 128
        n3 = 256
        fc1_input_size = n3 * (24**2)
        fc1_output = 128
        fc2_output = 64
        fc3_output = 32

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(3, n1, kernel_size=5,dilation=1,stride=1)
        self.conv2 = nn.Conv2d(n1,n2, kernel_size=5,dilation=2,stride=1)
        self.conv3 = nn.Conv2d(n2,n3, kernel_size=3,dilation=1,stride=1)

        self.fc1 = nn.Linear(fc1_input_size,fc1_output)
        self.fc2 = nn.Linear(fc1_output,fc2_output)
        self.fc3 = nn.Linear(fc2_output,fc3_output)
        self.out_layer = nn.Linear(fc3_output,num_classes)

        dropout_p = 0.5
        self.dropout = nn.Dropout(p=dropout_p)

        self.batch1 = nn.BatchNorm2d(n1,affine=True)
        self.batch2 = nn.BatchNorm2d(n2,affine=True)
        self.batch3 = nn.BatchNorm2d(n3,affine=True)

        self.fc_batch1 = nn.BatchNorm1d(fc1_output,affine=True)
        self.fc_batch2 = nn.BatchNorm1d(fc2_output,affine=True)
        self.fc_batch3 = nn.BatchNorm1d(fc3_output,affine=True)
        
    def forward(self, x):
        # (TODO) Forward the model
        # original forward
        # x = self.pool(self.relu(self.conv1(x)))
        # x = self.pool(self.relu(self.conv2(x)))
        # x = self.pool(self.relu(self.conv3(x)))
        # x = self.flatten(x)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        # x = self.out_layer(x)
        # return x
        
        # new forward
        x = self.pool(self.relu(self.batch1(self.conv1(x))))
        x = self.pool(self.relu(self.batch2(self.conv2(x))))
        x = self.pool(self.relu(self.batch3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc_batch1(self.fc1(x))))
        x = self.dropout(self.relu(self.fc_batch2(self.fc2(x))))
        x = self.dropout(self.relu(self.fc_batch3(self.fc3(x))))
        x = self.out_layer(x)
        return x

def train(model: nn.Module, train_loader: DataLoader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    sample_number = 0
    loop = tqdm(train_loader,desc="Traning",colour='#00CACA')
    for (image, label) in loop:
        image,label=image.to(device),label.to(device)
        batch_size = image.size(0)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
        sample_number += batch_size
    avg_loss = total_loss / sample_number
    return avg_loss


def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    model.eval()
    total_loss = 0.0
    total_correct = 0
    sample_number = 0
    with torch.no_grad():
        loop = tqdm(val_loader,desc="Validating")
        for (images,labels) in loop:
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            batch_size = images.size(0)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            sample_number += batch_size

    avg_loss = total_loss / sample_number
    accuracy = float(total_correct) / sample_number
    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, criterion, device):
    # (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
    model.eval()
    predictions = []
    ids = []
    index = 1
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing")
        for images, _ in loop:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            for p in preds:
                predictions.append(p)
                ids.append(index)
                index += 1
    # CSV
    with open('CNN.csv', mode='w', newline='') as f:
        f.write('id,prediction\n')
        for id,label in zip(ids,predictions):
            f.write(f"{id},{label}\n")

    print(f"Predictions saved to 'CNN.csv'")
    return