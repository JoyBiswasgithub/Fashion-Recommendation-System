# Fashion-Recommendation-System
## Feature Extraction using ResNet50, Search Similarity Using NearestNeighbors
# Fashion Recommendation System

This project implements a fashion image recommendation system using a pre-trained ResNet50 model for feature extraction and a k-Nearest Neighbors (k-NN) algorithm for retrieving similar images. The system is built using TensorFlow, Keras, and Streamlit for the web interface.

## Table of Contents
- [Project Overview](#project-overview)
- [Notebooks](#notebooks)
- [Scripts](#scripts)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Overview

The Fashion Recommendation System recommends similar fashion images based on a user's uploaded image. It uses a pre-trained ResNet50 model to extract image features, which are then used with a k-Nearest Neighbors (k-NN) algorithm to find and display similar images.

## Notebooks

This repository contains the following Jupyter Notebooks:

1. **`01_Data_Preparation.ipynb`**
   - Prepares and preprocesses the dataset. Includes data loading, transformation, and visualization of sample images.

2. **`02_Feature_Extraction.ipynb`**
   - Extracts features from images using a pre-trained ResNet50 model. Saves the features and filenames to pickle files.

3. **`03_Recommendation_System.ipynb`**
   - Implements the recommendation system using k-NN. Loads the features, trains the k-NN model, and integrates it with a Streamlit app for interactive recommendations.

## Scripts

### `feature_extraction.py`

Extracts features from images using a pre-trained ResNet50 model and saves them to pickle files.

- **Dependencies**: TensorFlow, NumPy, Pillow, scikit-learn, pickle
- **Usage**: 
  ```bash
  python feature_extraction.py
