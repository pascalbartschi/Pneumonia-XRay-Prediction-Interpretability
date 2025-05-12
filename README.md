# Pneumonia X-Ray Prediction and Interpretability

This project focuses on building a CNN classifier to predict pneumonia from chest X-ray images and applying interpretability techniques like Integrated Gradients and Grad-CAM to understand the model's predictions.

## Dataset

The dataset used is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download). Download the dataset and place it in the `data` folder.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Reproducing Results

1. Download the dataset and place it in the `data` folder.
2. Run `EDA.ipynb` to get some basic analytics of the dataset.
3. Run train_cnn.py which will create two state dictionaries named `cnn_model_randomized_with_sampling_1.pt` and `cnn_model_with_sampling_1.pt` which should be saved in a folder called `model_state_dicts` and give you accuracy and F1 score of the CNN run both with normal dataset and with a dataset where training samples have randomized labels. 
4. Finally run `attribution_maps_figure.ipynb` to generate all the images and plots. 
