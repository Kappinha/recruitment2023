import torch
import torchvision
from torch.utils.data import DataLoader
from bobnet import BobNet
import utils

training_set: torch.utils.data.Dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
validation_set: torch.utils.data.Dataset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

model1 = BobNet()

# batch size
MLP_BATCH_SIZE=64

# learning rate
MLP_LEARNING_RATE=0.001

# momentum
MLP_MOMENTUM=0.9

# training epochs to run
MLP_EPOCHS=10

# create the training loader
mlp_training_loader = DataLoader(training_set, batch_size=MLP_BATCH_SIZE, shuffle=True) 

# create the validation loader
mlp_validation_loader = DataLoader(validation_set, batch_size=MLP_BATCH_SIZE, shuffle=True)

mlp_loss_fn = torch.nn.CrossEntropyLoss()

mlp_optimizer = torch.optim.SGD(model1.parameters(), lr=MLP_LEARNING_RATE, momentum=MLP_MOMENTUM)

# how many batches between logs
LOGGING_INTERVAL=100

utils.train_model(model1, MLP_EPOCHS, mlp_optimizer, mlp_loss_fn, mlp_training_loader, mlp_validation_loader, LOGGING_INTERVAL)

##PART 2

from cnn import CNN

model2 = CNN()

mlp_optimizer_m2 = torch.optim.SGD(model2.parameters(), lr=MLP_LEARNING_RATE, momentum=MLP_MOMENTUM)

utils.train_model(model2, MLP_EPOCHS, mlp_optimizer_m2, mlp_loss_fn, mlp_training_loader, mlp_validation_loader, LOGGING_INTERVAL)


