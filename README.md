# APE

APE is **A** sim**P**le image classification fram**E**work for quick and easy training/evaluation/deployment.

## File structure

Codes are in the `ape` folder.

`train.py` is for training, `test.py` is for testing accuracy using test dataset, `predict.py` is for predicting the label of given images.

Configuration files are in `configs` folder. Configs are python files.

## Requirements

- Python>=3.6
- Pytorch>=1.1.0
- Tensorboard==1.14.0 (better with tensorflow installed, but not required)

## Quick start

### Dataset

Datasets are loaded using [torchvision.datasets.ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder). Simply arrange your dataset by categories, in this way:

```directories
$DATASET_PATH/dog/xxx.png
$DATASET_PATH/dog/xxy.png
$DATASET_PATH/dog/xxz.png

$DATASET_PATH/cat/123.png
$DATASET_PATH/cat/nsdf3.png
$DATASET_PATH/cat/asd932_.png
```

And it will be loaded automatically.

### Training

To train the model, first define a model config, `configs/model.py` for example. Then run:

```bash
python -u ape/train.py --config /path/to/your/config
# Example:
python -u ape/train.py --config configs/model.py
```

You can also resume a training process using a model checkpoint. Set `resume = True` and set the checkpoint path in the config, and it will be resumed from the checkpoint.

### Evaluation

To test the model using test dataset, run:

```bash
python -u ape/test.py --config /path/to/your/config
# Example:
python -u ape/test.py --config configs/model.py
```

### Inference

You can use `predict.py` to inference images using trained model. Run:

```bash
python -u ape/predict.py --config /path/to/your/config --path /path/to/your/images --ext <Extension-Name> --output /path/to/output/result
# Example:
python -u ape/predict.py --config configs/model.py --path /data/dataset/ --ext png --output infer_output/
```
