# type: ignore
import sys; sys.path.append('../')
import numpy as np
import urllib.request
import gzip
from tqdm import trange

epochs = 30

# Fetch data
def fetch(url, type_data = None):
    # Extract the dataset from the compressed file
    with gzip.open(urllib.request.urlopen(url)) as f:
        if type_data == 'label':
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        else:
            data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    return data

X_train_all = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
Y_train_all = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", 'label')
X_test_all = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
Y_test_all = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", 'label')

# Split data
n_train = 1000
n_test = 250

# Every row is a flattened image
X_train = X_train_all[:n_train].reshape(-1, 28*28).astype(np.float32)
Y_train = Y_train_all[:n_train].astype(np.float32)

X_test = X_test_all[:n_test].reshape(-1, 28*28).astype(np.float32)
Y_test = Y_test_all[:n_test].astype(np.float32)


######################
###### PyTORCH #######
######################

# from https://medium.com/@aungkyawmyint_26195/multi-layer-perceptron-mnist-pytorch-463f795b897a
import torch
import torch.nn as tnn
import torch.nn.functional as F

print('TORCH - SGD')
# define NN architecture
class Net(tnn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = tnn.Linear(28*28, 512, bias=False, dtype=torch.float32)
        self.fc2 = tnn.Linear(512,512, bias=False, dtype=torch.float32)
        self.fc3 = tnn.Linear(512,10, bias=False, dtype=torch.float32)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

torch_model = Net()
criterion = tnn.CrossEntropyLoss()
optimizer = torch.optim.SGD(torch_model.parameters(), 
	lr=0.001, momentum=0.9, nesterov=True)

torch_X_train = torch.from_numpy(X_train)
torch_Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)

# Save initial weights for giagrad test later
fc1_data = torch_model.fc1.weight.detach().numpy().copy()
fc2_data = torch_model.fc2.weight.detach().numpy().copy()
fc3_data = torch_model.fc3.weight.detach().numpy().copy()

for _ in (prog_bar := trange(epochs)):
    optimizer.zero_grad()
    output = torch_model(torch_X_train)
    tloss = criterion(output, torch_Y_train)
    tloss.backward()
    optimizer.step()
    prog_bar.set_description(f"loss: {tloss.data}")

######################
###### GIAGRAD #######
######################

import giagrad
import giagrad.nn as gnn

print('GIAGRAD - SGD')
class MLP(gnn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = gnn.Linear(28*28, 512, bias=False)
        self.fc1.w.data = fc1_data
        self.fc2 = gnn.Linear(512,512, bias=False)
        self.fc2.w.data = fc2_data
        self.fc3 = gnn.Linear(512,10, bias=False)
        self.fc3.w.data = fc3_data
        
    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x

gg_model = MLP()
criterion = gnn.CrossEntropyLoss()
optimizer = giagrad.optim.SGD(gg_model.parameters(), 
	lr=0.001, momentum=0.9, nesterov=True)

for _ in (prog_bar := trange(epochs)):
    optimizer.zero_grad()
    output = gg_model(X_train)
    gloss = criterion(output, Y_train.astype(np.int8))
    gloss.backward()
    optimizer.step()
    prog_bar.set_description(f"loss: {gloss.data}")


## TEST ##

tol = 1e-3

assert abs(tloss.item() - gloss.data) < tol 
assert np.all(abs(gg_model.fc1.w.data - torch_model.fc1.weight.detach().numpy()) < tol)
assert np.all(abs(gg_model.fc2.w.data - torch_model.fc2.weight.detach().numpy()) < tol)
assert np.all(abs(gg_model.fc3.w.data - torch_model.fc3.weight.detach().numpy()) < tol)


#### ADAM ####

print('TORCH - Adam')
# define NN architecture
class Net(tnn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = tnn.Linear(28*28, 512, bias=False, dtype=torch.float32)
        self.fc2 = tnn.Linear(512,512, bias=False, dtype=torch.float32)
        self.fc3 = tnn.Linear(512,10, bias=False, dtype=torch.float32)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

torch_model = Net()
criterion = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(torch_model.parameters(), 
	lr=0.001, betas=(0.9,0.999), eps=1e-8)

torch_X_train = torch.from_numpy(X_train)
torch_Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)
torch_X_test  = torch.from_numpy(X_test)
torch_Y_test  = torch.from_numpy(Y_test).type(torch.LongTensor)

# Save initial weights for giagrad test later
fc1_data = torch_model.fc1.weight.detach().numpy().copy()
fc2_data = torch_model.fc2.weight.detach().numpy().copy()
fc3_data = torch_model.fc3.weight.detach().numpy().copy()

for _ in (prog_bar := trange(epochs)):
    optimizer.zero_grad()
    output = torch_model(torch_X_train)
    tloss = criterion(output, torch_Y_train)
    tloss.backward()
    optimizer.step()
    prog_bar.set_description(f"loss: {tloss.data}")

import torch.nn.functional as F

print(f'train accuracy: {(F.softmax(torch_model(torch_X_train), dim=1).argmax(dim=1) == torch_Y_train).type(torch.FloatTensor).mean() * 100} %')
print(f'test accuracy: {(F.softmax(torch_model(torch_X_test),dim=1).argmax(dim=1) == torch_Y_test).type(torch.FloatTensor).mean() * 100} %')


######################
###### GIAGRAD #######
######################

import giagrad
import giagrad.nn as gnn

print('GIAGRAD - Adam')
class MLP(gnn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = gnn.Linear(28*28, 512, bias=False)
        self.fc1.w.data = fc1_data
        self.fc2 = gnn.Linear(512,512, bias=False)
        self.fc2.w.data = fc2_data
        self.fc3 = gnn.Linear(512,10, bias=False)
        self.fc3.w.data = fc3_data
        
    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x

gg_model = MLP()
criterion = gnn.CrossEntropyLoss()
optimizer = giagrad.optim.Adam(gg_model.parameters(), 
	lr=0.001, betas=(0.9,0.999), eps=1e-8)

for _ in (prog_bar := trange(epochs)):
    optimizer.zero_grad()
    output = gg_model(X_train)
    gloss = criterion(output, Y_train.astype(np.int8))
    gloss.backward()
    optimizer.step()
    prog_bar.set_description(f"loss: {gloss.data}")

gg_model.eval()

print(f'train accuracy: {(gg_model(X_train).softmax(axis=1).data.argmax(axis=1) == Y_train).mean() * 100} %')
print(f'test accuracy: {(gg_model(X_test).softmax(axis=1).data.argmax(axis=1) == Y_test).mean() * 100} %')
## TEST ##

tol = 5e-2 # 0.05

assert abs(tloss.item() - gloss.data) < tol 
assert np.all(abs(gg_model.fc1.w.data - torch_model.fc1.weight.detach().numpy()) < tol)
assert np.all(abs(gg_model.fc2.w.data - torch_model.fc2.weight.detach().numpy()) < tol)
assert np.all(abs(gg_model.fc3.w.data - torch_model.fc3.weight.detach().numpy()) < tol)


## ADADELTA ##

print('TORCH - Adadelta')
# define NN architecture
class Net(tnn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = tnn.Linear(28*28, 512, bias=False, dtype=torch.float32)
        self.fc2 = tnn.Linear(512,512, bias=False, dtype=torch.float32)
        self.fc3 = tnn.Linear(512,10, bias=False, dtype=torch.float32)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

torch_model = Net()
criterion = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(torch_model.parameters(), 
	lr=1.0, rho=0.9, eps=1e-6)

torch_X_train = torch.from_numpy(X_train)
torch_Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)
torch_X_test  = torch.from_numpy(X_test)
torch_Y_test  = torch.from_numpy(Y_test).type(torch.LongTensor)

# Save initial weights for giagrad test later
fc1_data = torch_model.fc1.weight.detach().numpy().copy()
fc2_data = torch_model.fc2.weight.detach().numpy().copy()
fc3_data = torch_model.fc3.weight.detach().numpy().copy()

for _ in (prog_bar := trange(epochs)):
    optimizer.zero_grad()
    output = torch_model(torch_X_train)
    tloss = criterion(output, torch_Y_train)
    tloss.backward()
    optimizer.step()
    prog_bar.set_description(f"loss: {tloss.data}")

import torch.nn.functional as F

print(f'train accuracy: {(F.softmax(torch_model(torch_X_train), dim=1).argmax(dim=1) == torch_Y_train).type(torch.FloatTensor).mean() * 100} %')
print(f'test accuracy: {(F.softmax(torch_model(torch_X_test),dim=1).argmax(dim=1) == torch_Y_test).type(torch.FloatTensor).mean() * 100} %')


######################
###### GIAGRAD #######
######################

import giagrad
import giagrad.nn as gnn

print('GIAGRAD - Adadelta')
class MLP(gnn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = gnn.Linear(28*28, 512, bias=False)
        self.fc1.w.data = fc1_data
        self.fc2 = gnn.Linear(512,512, bias=False)
        self.fc2.w.data = fc2_data
        self.fc3 = gnn.Linear(512,10, bias=False)
        self.fc3.w.data = fc3_data
        
    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x

gg_model = MLP()
criterion = gnn.CrossEntropyLoss()
optimizer = giagrad.optim.Adadelta(gg_model.parameters(), 
	lr=1.0, rho=0.9, eps=1e-6)

for _ in (prog_bar := trange(epochs)):
    optimizer.zero_grad()
    output = gg_model(X_train)
    gloss = criterion(output, Y_train.astype(np.int8))
    gloss.backward()
    optimizer.step()
    prog_bar.set_description(f"loss: {gloss.data}")

gg_model.eval()

print(f'train accuracy: {(gg_model(X_train).softmax(axis=1).data.argmax(axis=1) == Y_train).mean() * 100} %')
print(f'test accuracy: {(gg_model(X_test).softmax(axis=1).data.argmax(axis=1) == Y_test).mean() * 100} %')
## TEST ##

tol = 5e-2 # 0.05

assert abs(tloss.item() - gloss.data) < tol 
assert np.all(abs(gg_model.fc1.w.data - torch_model.fc1.weight.detach().numpy()) < tol)
assert np.all(abs(gg_model.fc2.w.data - torch_model.fc2.weight.detach().numpy()) < tol)
assert np.all(abs(gg_model.fc3.w.data - torch_model.fc3.weight.detach().numpy()) < tol)


## ADADELTA ##

print('TORCH - Adamax')
# define NN architecture
class Net(tnn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = tnn.Linear(28*28, 512, bias=False, dtype=torch.float32)
        self.fc2 = tnn.Linear(512,512, bias=False, dtype=torch.float32)
        self.fc3 = tnn.Linear(512,10, bias=False, dtype=torch.float32)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

torch_model = Net()
criterion = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(torch_model.parameters(), 
	lr=1e-3, betas=(0.9,0.999), eps=1e-8)

torch_X_train = torch.from_numpy(X_train)
torch_Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)
torch_X_test  = torch.from_numpy(X_test)
torch_Y_test  = torch.from_numpy(Y_test).type(torch.LongTensor)

# Save initial weights for giagrad test later
fc1_data = torch_model.fc1.weight.detach().numpy().copy()
fc2_data = torch_model.fc2.weight.detach().numpy().copy()
fc3_data = torch_model.fc3.weight.detach().numpy().copy()

for _ in (prog_bar := trange(epochs)):
    optimizer.zero_grad()
    output = torch_model(torch_X_train)
    tloss = criterion(output, torch_Y_train)
    tloss.backward()
    optimizer.step()
    prog_bar.set_description(f"loss: {tloss.data}")

import torch.nn.functional as F

print(f'train accuracy: {(F.softmax(torch_model(torch_X_train), dim=1).argmax(dim=1) == torch_Y_train).type(torch.FloatTensor).mean() * 100} %')
print(f'test accuracy: {(F.softmax(torch_model(torch_X_test),dim=1).argmax(dim=1) == torch_Y_test).type(torch.FloatTensor).mean() * 100} %')


######################
###### GIAGRAD #######
######################

import giagrad
import giagrad.nn as gnn

print('GIAGRAD - Adamax')
class MLP(gnn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = gnn.Linear(28*28, 512, bias=False)
        self.fc1.w.data = fc1_data
        self.fc2 = gnn.Linear(512,512, bias=False)
        self.fc2.w.data = fc2_data
        self.fc3 = gnn.Linear(512,10, bias=False)
        self.fc3.w.data = fc3_data
        
    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x

gg_model = MLP()
criterion = gnn.CrossEntropyLoss()
optimizer = giagrad.optim.Adamax(gg_model.parameters(), 
	lr=1e-3, betas=(0.9,0.999), eps=1e-8)

for _ in (prog_bar := trange(epochs)):
    optimizer.zero_grad()
    output = gg_model(X_train)
    gloss = criterion(output, Y_train.astype(np.int8))
    gloss.backward()
    optimizer.step()
    prog_bar.set_description(f"loss: {gloss.data}")

gg_model.eval()

print(f'train accuracy: {(gg_model(X_train).softmax(axis=1).data.argmax(axis=1) == Y_train).mean() * 100} %')
print(f'test accuracy: {(gg_model(X_test).softmax(axis=1).data.argmax(axis=1) == Y_test).mean() * 100} %')

## TEST ##

tol = 5e-2 # 0.05

assert abs(tloss.item() - gloss.data) < tol 
assert np.all(abs(gg_model.fc1.w.data - torch_model.fc1.weight.detach().numpy()) < tol)
assert np.all(abs(gg_model.fc2.w.data - torch_model.fc2.weight.detach().numpy()) < tol)
assert np.all(abs(gg_model.fc3.w.data - torch_model.fc3.weight.detach().numpy()) < tol)
