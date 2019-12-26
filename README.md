# APE

APE is **A** sim**P**le image classification fram**E**work for quick and easy training/validation/deployment.

## Training Codes

Training codes are in `tools` folder.

`train.py` is for training, `test.py` is for testing accuracy using test dataset, `predict.py` is for predicting the label of some images.

## How to run

Environment prerequisites:

    1. Python>=3.6
    2. Pytorch>=1.1.0
    3. Tensorboard==1.14.0 (better with tensorflow installed, but not required)

To train the model, define a model with the template `configs/model.json`. Run `tools/train.py --model=configs/model.json`.

You can also resume training using the model checkpoint, just set `resume = True` and set the model dir/name.

To test the model using test set, run `tools/test.py --model=configs/model.json`.
