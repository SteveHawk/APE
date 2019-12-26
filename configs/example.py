from torch import nn
import sys
sys.path.append('../')
from train import Params, View, train
from test import test

if __name__ == "__main__":
    params = Params(
        # Define the model structure
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=2, dilation=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=2, dilation=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            View(),
        ),
        # How many epochs
        epochs = 50,
        # Bach size
        bs = 64,
        # Learning rate
        lr = 0.1,
        # The target accuracy to stop the training
        target_acc = 96,
        # Where to store/load model checkpoints
        model_path = "./models/",
        # Location of dataset
        data_path = "../dataset/",
        # What size should pics be preprocessed
        transform_img_size_x = 256,
        transform_img_size_y = 256,
        # If resume training
        resume = False,
        # The name of that model
        resume_model_name = "model_checkpoint_max_acc.pth.tar",
        # Output info control
        verbose = [True, True, True, True, True],  # train_loss, valid_loss, train_acc, valid_acc, epoch progress bar
        # Which device to use (None for cpu)
        dev_num = None,
        # How many processes / threads for dataloader. 0 means working in main process.
        num_workers = 0,
        #Location of test dataset
        test_data_path = "../test_dataset/",
        # Name of the test model
        test_model_name = "model_checkpoint_max_acc.pth.tar"
    )

    # Choose mode
    TRAIN = True
    if TRAIN:
        train(params)
    else:
        test(params)
