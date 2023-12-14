import torch
import torchvision.datasets as datasets
import torch.nn as  nn
from torch.utils.data import DataLoader
from torch import optim
import torchvision.transforms as transforms
from utils import accuracy,dequeue
from model import ResNet50,ResNet101,ResNet152
from tqdm import tqdm
import numpy as np



cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
cifar_trainset,cifar_valset = torch.utils.data.random_split(cifar_trainset, [45000, 5000])
train_loader = DataLoader(dataset=cifar_trainset, batch_size=256, shuffle=True)
val_loader = DataLoader(dataset=cifar_valset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=cifar_testset, batch_size=256, shuffle=True)

device=  torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet101().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

val_accs = dequeue(10)

for epoch in range(100):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        

        data_copy = data.detach().clone()

        data = data.to(device=device)
        data_copy = data_copy.to(device=device)
        targets = targets.to(device=device)

        layers_1 = {}
        layers_2 = {}

        for i in range(1,17): # 16 for ResNet50, 33 for ResNet101, 50 for ResNet152 
           layers_1[i] = np.random.randint(0,2)
           layers_2[i] = np.random.randint(0,2)

        scores_1 = model([data,layers_1,1])
        scores_1_copy = scores_1.detach().clone()
        scores_2 = model([data_copy,layers_2,1])
        scores_2_copy = scores_2.detach().clone()

        loss = (criterion(scores_1, targets) + criterion(scores_2, targets))/2

        loss += 0.3 * (criterion(scores_1, scores_2_copy) + criterion(scores_1_copy, scores_1))/2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_acc = float(accuracy(val_loader,model).cpu().numpy())
    print("validation accuracy: ",val_acc)
    if(epoch>10 and val_acc < val_accs.average()):
      break;
    val_accs.add(val_acc)

