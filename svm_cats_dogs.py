import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# CONFIG
IMG_SIZE = 64
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
MAX_TRAIN_IMAGES = 5000   # LIMIT FOR SPEED

# HOG FEATURE FUNCTION
def extract_hog(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

# LOAD TRAINING DATA
X = []
y = []

import random

all_files = os.listdir(TRAIN_DIR)
random.shuffle(all_files)
train_files = all_files[:MAX_TRAIN_IMAGES]


for img_name in tqdm(train_files, desc="Loading training images"):
    path = os.path.join(TRAIN_DIR, img_name)

    # Label
    label = 0 if "cat" in img_name else 1

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # SAFETY CHECK
    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    features = extract_hog(img)

    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Total training samples:", len(X))

# TRAIN / VALIDATION SPLIT
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# FEATURE SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# TRAIN SVM
svm = SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)

print("Training SVM...")
svm.fit(X_train, y_train)

# VALIDATION
y_pred = svm.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", accuracy)

# TRAIN ON FULL DATA
X = scaler.fit_transform(X)
svm.fit(X, y)

# PREDICT TEST DATA
ids = []
labels = []

test_files = sorted(os.listdir(TEST_DIR))

for img_name in tqdm(test_files, desc="Predicting test images"):
    img_id = int(img_name.split('.')[0])
    path = os.path.join(TEST_DIR, img_name)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    features = extract_hog(img).reshape(1, -1)
    features = scaler.transform(features)

    pred = svm.predict(features)[0]

    ids.append(img_id)
    labels.append(pred)

# SAVE SUBMISSION
submission = pd.DataFrame({
    "id": ids,
    "label": labels
})

submission.to_csv("submission.csv", index=False)
print("submission.csv generated successfully!")
# PREDICT SINGLE IMAGE

def predict_single_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not found!")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    features = extract_hog(img).reshape(1, -1)
    features = scaler.transform(features)

    prediction = svm.predict(features)[0]

    if prediction == 0:
        print("Prediction: CAT 🐱")
    else:
        print("Prediction: DOG 🐶")


# Example usage
predict_single_image("predict/my_image.jpeg")
