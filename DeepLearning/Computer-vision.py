import os
import timeit
import time
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pformat
from tqdm import tqdm
from google.colab import drive
import sys
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import *
from torchvision import transforms, datasets

torch.multiprocessing.set_sharing_strategy('file_system')
cudnn.benchmark = True

# Define original model from scratch 
# else use pretrained model 
# reference : https://pytorch.org/docs/stable/torchvision/models.html
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()  

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
    torch.nn.init.xavier_uniform_(self.conv1.weight)
    self.fc1 = nn.Linear(in_features = 256 * 8 * 8, out_features=1024)
    torch.nn.init.xavier_uniform_(self.fc1.weight)


  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x))) 
    x = self.Dropout(x)
    return x


# pre trained models
"""
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)
"""

def load_data(config):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # download data / use kaggle input
    df = pd.read_csv('/input',index = 'id')
    target = pd.DataFrame(df['target'])
    del df['target']
    sub = pd.read_csv('/input',index = 'id')
    tested = pd.DataFrame(df['target'])

    train = data_utils.TensorDataset(torch.Tensor(np.array(df)), torch.Tensor(np.array(target)))
    train_set, val_set = torch.utils.data.random_split(train, [50000, 10000])
    test = data_utils.TensorDataset(torch.Tensor(np.array(sub)), torch.Tensor(np.array(target)))

    train_dataloader  = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    valid_dataloader  = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True, num_workers=0)
   
    return train_dataloader, valid_dataloader, test_dataloader

def train(trainloader, validloader, device, config):
    log_interval = 100
    correct = 0
    validation_loss = 0
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['num_epochs']):   
      if epoch <= 10:
          optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['regular_constant'])
      elif epoch > 10 and epoch < 50:
          optimizer = torch.optim.Adam(model.parameters(), lr=config['lr']/10, weight_decay=config['regular_constant']) 
      else:
          optimizer = torch.optim.Adam(model.parameters(), lr=config['lr']/50, weight_decay=config['regular_constant']) 
      model.train()
      for batch_index, (images, labels) in enumerate(trainloader): 
          images = images.to(device) 
          labels = labels.to(device)
          optimizer.zero_grad()
          output = model(images)
          loss = criterion(output, labels) 
          loss.backward()
          optimizer.step()
          if (batch_index % log_interval) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_index * len(images), len(trainloader.dataset),
            100. * batch_index / len(trainloader), loss.item()))

      with torch.no_grad():
        validation_loss = 0
        correct = 0
        for images, labels in validloader:
          images = images.to(device)
          labels = labels.to(device)
          output = model(images)
          index, predictions = torch.max(output.data, 1)

          validation_loss += F.nll_loss(output, labels, size_average=False).item()
          correct += (predictions == labels).sum().item()      
        validation_loss /= len(validloader.dataset)
        print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(validloader.dataset),
        100. * correct / len(validloader.dataset)))
                  
    return model

# else use pretrained model evaluation mode

def test(net, testloader, device):
  predictions = []
  correct = 0
  total = 0
  net.eval()
  with torch.no_grad():
      for images, labels in testloader:
          images = images.to(device)
          labels = labels.to(device)
          outputs = net(images)
          indices, pred = torch.max(outputs.data, 1)  
          predictions.append(pred)
          correct += (pred == labels).sum().item()
          total += labels.size(0)
  accuracy = 100. * correct / total

  return accuracy, correct, total, predictions

def run():
  config = {
        'lr': 1e-3,
        'num_epochs': 10,
        'batch_size': 64,
        'num_classes': 10,
        'momentum':0.5,
        'regular_constant': 5e-4,
       }
    
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  train_dataloader, valid_dataloader, test_dataloader = load_data(config)
  
  model = train(train_dataloader, valid_dataloader, device, config)
  
  # Testing and saving for submission
  device = torch.device("cpu")

  assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
  checkpoint = torch.load('./checkpoint/ckpt.pth')
  model.load_state_dict(checkpoint)
  model.eval()
  
  start_time = timeit.default_timer()
  test_acc, test_correct, test_total,preds = test(model.to(device), test_dataloader, device)
  end_time = timeit.default_timer()
  test_time = (end_time - start_time)
  
  return test_acc, test_correct, test_time,preds

def compute_score(acc, min_thres=65, max_thres=8):
  if acc <= min_thres:
      base_score = 0.0
  elif acc >= max_thres:
      base_score = 100.0
  else:
      base_score = float(acc - min_thres) / (max_thres - min_thres) * 100
  return base_score

def main():
    
    accuracy, correct, run_time,preds = run()
    
    score = compute_score(accuracy)
    
    result = OrderedDict(correct=correct,
                         accuracy=accuracy,
                         run_time=run_time,
                         score=score)
  
    with open('result.txt', 'w') as f:
        f.writelines(pformat(result, indent=4))
    print("\nResult:\n", pformat(result, indent=4))
    sub = pd.read_csv('/input',index = 'id')
    sub['target'] = preds
    sub.to_csv('submission.csv')

main()
