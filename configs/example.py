from torch import nn
from ape.utils.view import View
from ape.utils.configs import Configs

# Number of classes
Configs.num_classes = 2

# Use gray scale images or color images
Configs.gray_scale = True

# Define the model structure (StevNet as example)
Configs.model = nn.Sequential(
    nn.Conv2d(1 if Configs.gray_scale else 3, 69, kernel_size=3, stride=1, padding=1, dilation=1),
    nn.BatchNorm2d(69),
    nn.ReLU(),
    nn.Conv2d(69, 69, kernel_size=3, stride=1, padding=2, dilation=2),
    nn.AvgPool2d(2),
    nn.BatchNorm2d(69),
    nn.ReLU(),
    nn.Conv2d(69, 69, kernel_size=3, stride=1, padding=4, dilation=4),
    nn.AvgPool2d(2),
    nn.BatchNorm2d(69),
    nn.ReLU(),
    nn.Conv2d(69, 69, kernel_size=3, stride=1, padding=8, dilation=8),
    nn.AvgPool2d(2),
    nn.BatchNorm2d(69),
    nn.ReLU(),
    nn.Conv2d(69, 32, kernel_size=3, stride=1, padding=1, dilation=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, Configs.num_classes, kernel_size=3, stride=1, padding=1, dilation=1),
    nn.BatchNorm2d(Configs.num_classes),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    View(),
)

# How many epochs
Configs.num_epochs = 50

# Bach size
Configs.bs = 64

# Learning rate
Configs.lr = 0.1

# The target accuracy to stop the training
Configs.target_acc = 96

# Where to store/load model checkpoints
Configs.model_path = "./models/"

# Location of dataset
Configs.data_path = "/path/to/the/dataset/"

# mean and std of the dataset (should be calculated based on your dataset)
# if three channel, use something like [0.5, 0.5, 0.5]
Configs.ds_mean = [0.5]
Configs.ds_std = [0.5]

# What size should pics be preprocessed
Configs.img_size_x = 256
Configs.img_size_y = 256

# If resume training
Configs.resume = False

# The name of the model for resuming
Configs.resume_model_name = "model_checkpoint_max_acc.pth.tar"

# Output info control
Configs.verbose = [True, True, True, True, True]  # train_loss, valid_loss, train_acc, valid_acc, epoch progress bar

# Which device for computing (None for cpu)
Configs.dev_num = None

# How many processes/threads for dataloader. 0 means working in main process.
Configs.num_workers = 0

# Location of test dataset
Configs.test_data_path = "/path/to/the/test_dataset/"

# Name of the model to be tested
Configs.test_model_name = "model_checkpoint_max_acc.pth.tar"

# Name of the model for prediction
Configs.prediction_model_name = "model_checkpoint_max_acc.pth.tar"
