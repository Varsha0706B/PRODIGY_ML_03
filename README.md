# Cats vs Dogs Classification using SVM

This project implements a Support Vector Machine (SVM) model to classify images of cats and dogs using HOG feature extraction.

## Dataset
Kaggle Dogs vs Cats Dataset  
https://www.kaggle.com/c/dogs-vs-cats

## Approach
- Image preprocessing (grayscale, resize)
- HOG feature extraction
- SVM (RBF kernel)
- Subset training (5000 images)

## Technologies
- Python
- OpenCV
- Scikit-learn
- Scikit-image

## How to Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python svm_cats_dogs.py
