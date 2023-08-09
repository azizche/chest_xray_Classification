# Chest Xray Classification

A deep learning approach that reads XRay chest images and predicts whether the vest is affected by pneumonia or not using PyTorch.

## Pneumonia vs Normal Chest Xray

### Pneumonia

<img "https://github.com/azizche/chest_xray_Classification/blob/main/Images/pneumonia.png" width=300 height= 300/>

### Normal

<img "https://github.com/azizche/chest_xray_Classification/blob/main/Images/normal.png" width=300 height= 300/>

## Best model parameters

| Property           | Values               |
| ------------------ | -------------------- |
| Pretrained Model   | Alexnet              |
| Optimizer used     | Adam optimizer       |
| Loss function Used | Binary Cross Entropy |
| Learning rate      | 0.01                 |
| Mini Batch Size    | 16                   |
| Epochs             | 10                   |
| Seed               | 1                    |

## Evaluation

<img "https://github.com/azizche/chest_xray_Classification/blob/main/Images/confusion_matrix.png" width=300 height= 300/>

### Tensorboard accuracy tracking

<img "https://github.com/azizche/chest_xray_Classification/blob/main/Images/tensorboard_visualisation.png" width=300 height= 300/>
** Test Accuracy: 96.62 %**

## DataSet

The dataset is from huggingspace [chest-xray-classification](https://huggingface.co/datasets/keremberke/chest-xray-classification). It has up to 5 820 images of labeled Xray Chest images.
The data is downloaded within the approach so you don't have to download it yourself.

## How to use?

The above code can be used for Deep Transfer Learning on any Image dataset to train using whether Alexnet or Efficient Net B7 as the PreTrained network.

### Steps to follow

1. Run any model you want that are available (currently Alexnet and Efficient Net B7 are available) and choose the hyperparameters you want

`python train.py --model_name <MODEL NAME> --batch_size <BATCH SIZE> --lr <Learning Rate> --epochs <EPOCHS> --seed <SEED>`

2. A folder called runs will be created in your directory. It's an ouput created bu the SummaryWriter instance of Tensorboard that saves the train/test accuracy/loss values.
   To visualise these values, run `%load_ext tensorboard` and then `!tensorboard --logdir runs"`.

## Some Prediction

<img "https://github.com/azizche/chest_xray_Classification/blob/main/Images/predictions.png" width=300 height= 300/>

## Deployed model

Make sure to check out the application that I've created in huggingspace using Gradio by clicking [here](https://huggingface.co/spaces/Aziizzz/ChestXrayClassification). You can upload any ChestXray image you like and check out what the model predicts!

## Contribute

If you want to contribute and add new feature feel free to send Pull request [here](https://github.com/azizche/chest_xray_Classification/pulls)

To report any bugs or request new features, head over to the [Issues page](https://github.com/azizche/chest_xray_Classification/issues)

## To-Do

- [ ] Add loggers functionality.
- [ ] Add more models and find better accuracy.
- [ ] Add Data augmentation.
