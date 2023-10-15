import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) # Criacao da primeira convolutional layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Criacao da segunda convolutional layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        x = torch.relu(self.conv1(x)) # Aplica a primeira convolutional layer
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x)) # Aplica a segunda convolutional layer
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = torch.relu(self.fc1(x)) #Aplica a primeira camada fc
        x = torch.relu(self.fc2(x)) # 2 camada fc
        x = self.fc3(x) # esta camada produz o output
        return x
