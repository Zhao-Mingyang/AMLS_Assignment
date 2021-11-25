import torch.nn.functional as F
import torch.nn as nn
import torch

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=1, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        
       
        # max pooling layer
        self.pool1 = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((2, 2),2)
        #linear layer (512 -> 2)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 4)

        self.dropout1 = nn.Dropout(0.5)
#         self.dropout2 = nn.Dropout(0.2)
#         self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x =  F.relu(self.conv1(x))
        x = self.pool2(self.dropout1(x))
        x = F.relu(self.conv2(x))
        x = self.pool2(self.dropout1(x))
        x =  F.relu(self.conv3(x))
        x = self.pool2(self.dropout1(x))
        x =  F.relu(self.conv4(x))
        x = self.pool2(self.dropout1(x))

        # flatten image input
        x = torch.flatten(x, 1)
#         x = x.reshape(8,-1)
#         print(x.size())
#         torch.reshape(x, (-1,2048))
        # add dropout layer
        
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
#         x = self.dropout3(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x

# create a complete CNN
model = Net()