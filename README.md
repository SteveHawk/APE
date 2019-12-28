# APE

APE is **A** sim**P**le image classification fram**E**work for quick and easy training/validation/deployment.

## Training Codes

Training codes are in `tools` folder.

`train.py` is for training, `test.py` is for testing accuracy using test dataset, `predict.py` is for predicting the label of some images.

## How to run

Environment prerequisites:

```requirements
Python>=3.6
Pytorch>=1.1.0
Tensorboard==1.14.0 (better with tensorflow installed, but not required)
```

Before running, you should set the `PYTHONPATH` environment variable to the root of this project folder. Use `$env:PYTHONPATH="."` in PowerShell, or `export PYTHONPATH=$PYTHONPATH:.` in Bash.

To train the model, first define a model config, `configs/model.py` for example. Then run `python -u tools/train.py --config configs/model.py`.

You can also resume training using the model checkpoint, just set `resume = True` and set the model dir/name.

To test the model using test set, run `python -u tools/test.py --config <YOUR-CONFIG-LOCATION>`.
