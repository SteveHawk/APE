from torch import nn
from tools.train import View


class Params:
    # Number of classes
    num_classes = 2

    # Use gray scale images or color images
    gray_scale = True

    # Define the model structure (StevNet as example)
    model = nn.Sequential(
        nn.Conv2d(1 if gray_scale else 3, 69, kernel_size=3, stride=1, padding=1, dilation=1),
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
        nn.Conv2d(32, num_classes, kernel_size=3, stride=1, padding=1, dilation=1),
        nn.BatchNorm2d(num_classes),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        View(),
    )

    # How many epochs
    num_epochs = 50

    # Bach size
    bs = 64

    # Learning rate
    lr = 0.1

    # The target accuracy to stop the training
    target_acc = 96

    # Where to store/load model checkpoints
    model_path = "./models/"

    # Location of dataset
    data_path = "/path/to/the/dataset/"

    # mean and std of the dataset
    ds_mean = 0.5
    ds_std = 0.5

    # What size should pics be preprocessed
    transform_img_size_x = 256
    transform_img_size_y = 256

    # If resume training
    resume = False

    # The name of the model for resuming
    resume_model_name = "model_checkpoint_max_acc.pth.tar"

    # Output info control
    verbose = [True, True, True, True, True]  # train_loss, valid_loss, train_acc, valid_acc, epoch progress bar

    # Which device for computing (None for cpu)
    dev_num = None

    # How many processes/threads for dataloader. 0 means working in main process.
    num_workers = 0

    # Location of test dataset
    test_data_path = "/path/to/the/test_dataset/"

    # Name of the model to be tested
    test_model_name = "model_checkpoint_max_acc.pth.tar"
