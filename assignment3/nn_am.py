
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    img = img.resize((64,64))
    data = np.asarray(img, dtype="int32")
    return data

def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(
        np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class GtsrbDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.data = [(X[i], y[i]) for i in range(len(X))]
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.data[idx]

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)

class Basic_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12288, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


zeros = []
ones = []

data_dirs = ["../data/Train/0", "../data/Train/1"]

print("Beginning image reading")
for file in os.listdir(data_dirs[0]):
    #print("Reading: {}".format(file))
    zeros.append(load_image(os.path.join(data_dirs[0],file)).flatten())

for file in os.listdir(data_dirs[1]):
    #print("Reading: {}".format(file))
    ones.append(load_image(os.path.join(data_dirs[1],file)).flatten())

# creating np arrays for data
X = np.array(zeros + ones)
zero_labels = np.zeros((len(zeros),1))
one_labels = np.ones((len(ones),1))
y = np.ravel(np.vstack((zero_labels, one_labels)))

# creating train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# casting np arrays to tensors
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

# creating data loaders for data
device = get_default_device()
train_data = GtsrbDataset(X_train, y_train)
train_data = DeviceDataLoader(torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True), device)
test_data = GtsrbDataset(X_test, y_test)
test_data = DeviceDataLoader(torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True), device)

# creating NN and putting it on the GPU
net = Basic_Net()
net = net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
EPOCHS = 100

# training the nn
print("Training the net...")
for epoch in range(EPOCHS):
    for data in train_data:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 12288))
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0

# evaluating nn performance
with torch.no_grad():
    for data in test_data:
        X, y = data
        output = net(X.view(-1,12288))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))




