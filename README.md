# Facial Expression Recognition

This repository contains an implementation of facial expression recognition using deep learning. The project compares the performance of two pre-trained models (VGG16 and ResNet50) on the FER2013 dataset, which contains images representing seven different facial emotions.

## Overview

Facial expression recognition is a computer vision task that involves identifying human emotions from facial images. This implementation uses transfer learning with popular convolutional neural network architectures to classify facial expressions into seven categories:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Features

- Transfer learning using VGG16 and ResNet50 pre-trained models
- Data augmentation to improve model generalization
- Comprehensive model evaluation with accuracy, classification reports, and confusion matrices
- Performance comparison between different CNN architectures
- Visualization of training and validation metrics

## Requirements

- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Dataset

This project uses the FER2013 dataset, which contains 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.

The dataset should be organized as follows:
```
/kaggle/input/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## Usage

1. Ensure you have the required dependencies installed
2. Set up the dataset in the expected directory structure
3. Run the script to train and evaluate both models:

```python
python facial_expression_recognition.py
```

## Model Architecture

Both VGG16 and ResNet50 models are used with their pre-trained weights on ImageNet. The base models are frozen and used as feature extractors. On top of these base models, we add:

1. Global Average Pooling
2. A dense layer with 128 units and ReLU activation
3. Dropout for regularization
4. A final dense layer with softmax activation for 7-class classification

## Results

The script outputs:
- Test accuracy for both models
- Detailed classification reports showing precision, recall, and F1-scores for each emotion
- Confusion matrices for visual inspection of classification performance
- Comparative plots of validation accuracy and loss

## Future Improvements

- Fine-tuning the pre-trained models
- Exploring more advanced architectures like EfficientNet or Vision Transformers
- Implementing ensemble methods to improve overall accuracy
- Adding real-time facial expression recognition using webcam input

