# %%
import cuml
from cuml.ensemble import RandomForestClassifier as cuMLRandomForestClassifier
from cuml.metrics import accuracy_score as cuml_accuracy_score, f1_score as cuml_f1_score, recall_score as cuml_recall_score
import numpy as np
import pandas as pd
import cupy as cp
from skimage.feature import hog
import joblib
from collections import Counter
import cv2
from cuml.model_selection import train_test_split

height, width = 10, 10

# %%
data = pd.read_csv(f"../archive/ascii_character_classification_{height}_x_{width}.csv", header=0).sample(frac=.05)
label_counts = Counter(data.iloc[:, 0])
print(label_counts)


# %%
# Move to GPU by converting to cupy arrays (cuML uses GPU arrays via CuPy)
X = cp.array(data.iloc[:, 1:].astype("float64"))
y = cp.array(data.iloc[:, 0].astype("float64"))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# cuML Random Forest Classifier
clf = cuMLRandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model
joblib.dump(clf, '../artifacts/random_forest_model_10_x_10_cuml.pkl')


# %%
# Predict and evaluate
y_pred_train = clf.predict(X_train)
train_accuracy = cuml_accuracy_score(y_train, y_pred_train)

y_pred_test = clf.predict(X_test)
test_accuracy = cuml_accuracy_score(y_test, y_pred_test)

f1 = cuml_f1_score(y_test, y_pred_test, average='weighted')
recall = cuml_recall_score(y_test, y_pred_test, average='weighted')

print(f"Train Accuracy: {train_accuracy*100:.4f}%")
print(f"Test Accuracy: {test_accuracy*100:.4f}%")
print(f"F1 Score: {f1*100:.4f}%")
print(f"Recall: {recall*100:.4f}%")

# %%
def extract_hog_features(images):
    hog_features = []
    for image in images:
        image_reshaped = image.reshape((height, width))
        features = hog(image_reshaped, pixels_per_cell=(2, 2), cells_per_block=(1, 1), feature_vector=True)
        hog_features.append(features)
    return np.array(hog_features)

# Move HOG features extraction to GPU (optional if needed)
X_hog = extract_hog_features(cp.asnumpy(X))  # Convert to CPU to use scikit-image
X_hog = cp.array(X_hog)  # Convert back to GPU

X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=42)


# %%
# cuML Random Forest Classifier for HOG features
clf_hog = cuMLRandomForestClassifier(n_estimators=100, random_state=42)
clf_hog.fit(X_train, y_train)

# Save the HOG model
joblib.dump(clf_hog, '../artifacts/random_forest_model_hog_10_x_10_cuml.pkl')


# %%
y_pred_train_hog = clf_hog.predict(X_train)
train_accuracy_hog = cuml_accuracy_score(y_train, y_pred_train_hog)

y_pred_test_hog = clf_hog.predict(X_test)
test_accuracy_hog = cuml_accuracy_score(y_test, y_pred_test_hog)

f1_hog = cuml_f1_score(y_test, y_pred_test_hog, average='weighted')
recall_hog = cuml_recall_score(y_test, y_pred_test_hog, average='weighted')

print(f"Train Accuracy (HOG): {train_accuracy_hog*100:.4f}%")
print(f"Test Accuracy (HOG): {test_accuracy_hog*100:.4f}%")
print(f"F1 Score (HOG): {f1_hog*100:.4f}%")
print(f"Recall (HOG): {recall_hog*100:.4f}%")


# %%
# Extract SIFT features
def extract_sift_features(images):
    sift = cv2.SIFT_create()
    sift_features = []
    
    for image in images:
        image_reshaped = image.reshape((height, width)).astype(np.uint8)
        keypoints, descriptors = sift.detectAndCompute(image_reshaped, None)
        
        if descriptors is None:
            descriptors = np.zeros((1, sift.descriptorSize()), dtype=np.float32)
        
        features = descriptors.flatten()
        sift_features.append(features)
    
    return np.array(sift_features)

X_sift = extract_sift_features(cp.asnumpy(X))  # Convert to CPU for OpenCV
X_sift = cp.array(X_sift)  # Convert back to GPU

# Pad descriptors to consistent length
max_len = max(len(f) for f in X_sift)
X_sift = cp.array([np.pad(f, (0, max_len - len(f)), 'constant') for f in X_sift])

X_train, X_test, y_train, y_test = train_test_split(X_sift, y, test_size=0.2, random_state=42)


# %%
# cuML Random Forest Classifier for SIFT features
clf_sift = cuMLRandomForestClassifier(n_estimators=100, random_state=42)
clf_sift.fit(X_train, y_train)

# Save the SIFT model
joblib.dump(clf_sift, '../artifacts/random_forest_model_sift_10_x_10_cuml.pkl')

# %%
y_pred_train_sift = clf_sift.predict(X_train)
train_accuracy_sift = cuml_accuracy_score(y_train, y_pred_train_sift)

y_pred_test_sift = clf_sift.predict(X_test)
test_accuracy_sift = cuml_accuracy_score(y_test, y_pred_test_sift)

f1_sift = cuml_f1_score(y_test, y_pred_test_sift, average='weighted')
recall_sift = cuml_recall_score(y_test, y_pred_test_sift, average='weighted')

print(f"Train Accuracy (SIFT): {train_accuracy_sift*100:.4f}%")
print(f"Test Accuracy (SIFT): {test_accuracy_sift*100:.4f}%")
print(f"F1 Score (SIFT): {f1_sift*100:.4f}%")
print(f"Recall (SIFT): {recall_sift*100:.4f}%")


