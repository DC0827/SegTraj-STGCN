import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils import *
from models_solera import *

folder = "../../data/Sim"  # students03   NBA  Hotel  ETH  Sim
train_path = os.path.join(folder, "examples_train_unnormalized.pkl")
print("train_path: ", train_path)
valid_path = os.path.join(folder, "examples_valid_unnormalized.pkl")
print("valid_path: ", valid_path)
test_path = os.path.join(folder, "examples_test_unnormalized.pkl")
print("test_path: ", test_path)

features_train_path = os.path.join(folder, "pairwise_features_train.pkl")
print("features train_path: ", features_train_path)
features_valid_path = os.path.join(folder, "pairwise_features_valid.pkl")
print("features valid_path: ", features_valid_path)
features_test_path = os.path.join(folder, "pairwise_features_test.pkl")
print("features test_path: ", features_test_path)

train_label_path = os.path.join(folder, "clusters_train.pkl")
valid_label_path = os.path.join(folder, "clusters_valid.pkl")
test_label_path = os.path.join(folder, "clusters_test.pkl")
print("train label path: ", train_label_path)
print("valid label path: ", valid_label_path)
print("test label path: ", test_label_path)

# Load training, validation and test data
with open(train_path, 'rb') as f:
    examples_train = pickle.load(f)
with open(valid_path, 'rb') as f:
    examples_valid = pickle.load(f)
with open(test_path, 'rb') as f:
    examples_test = pickle.load(f)

# Load pairwise features
with open(features_train_path, 'rb') as f:
    features_train = pickle.load(f)
with open(features_valid_path, 'rb') as f:
    features_valid = pickle.load(f)
with open(features_test_path, 'rb') as f:
    features_test = pickle.load(f)

# Load clusters (labels)
with open(train_label_path, 'rb') as f:
    clusters_train = pickle.load(f)
with open(valid_label_path, 'rb') as f:
    clusters_valid = pickle.load(f)
with open(test_label_path, 'rb') as f:
    clusters_test = pickle.load(f)

# Build ground and train model
ground = build_ground(examples_train)
ssvm = SoleraSVM(n_features=6)
ssvm.fit(examples_train, ground, clusters_train, n_iters=50000, verbose=1, 
         verbose_iters=100, pairwise_features=features_train)


# Evaluate on test set
recalls, precisions, F1s, ACC = [], [], [],[]
for i in range(len(examples_test)):
    example = examples_test[i]
    label = clusters_test[i]
    features = features_test[i]
    predicted, _ = ssvm.predict(example, ground, features)
    recall, precision, F1 = compute_groupMitre(label, predicted)
    acc = compute_pairwise_accuracy(label, predicted)
    recalls.append(recall)
    precisions.append(precision)
    F1s.append(F1)
    ACC.append(acc)
print("Test set:")
print("Average precision: ", np.mean(precisions))
print("Average recall: ", np.mean(recalls))
print("Average F1: ", np.mean(F1s))
print(f"Average ACC: {np.mean(ACC)}%")

# Evaluate on validation set
recalls, precisions, F1s , ACC = [], [], [] ,[]
for i in range(len(examples_valid)):
    example = examples_valid[i]
    label = clusters_valid[i]
    features = features_valid[i]
    predicted, _ = ssvm.predict(example, ground, features)
    recall, precision, F1 = compute_groupMitre(label, predicted)
    acc = compute_pairwise_accuracy(label, predicted)
    recalls.append(recall)
    precisions.append(precision)
    F1s.append(F1)
    ACC.append(acc)
print("Validation set:")
print("Average precision: ", np.mean(precisions))
print("Average recall: ", np.mean(recalls))
print("Average F1: ", np.mean(F1s))
print(f"Average ACC: {np.mean(ACC)}%")

# Evaluate on training set
recalls, precisions, F1s ,ACC = [], [], [] ,[]
for i in range(len(examples_train)):
    example = examples_train[i]
    label = clusters_train[i]
    features = features_train[i]
    predicted, _ = ssvm.predict(example, ground, features)
    recall, precision, F1 = compute_groupMitre(label, predicted)
    acc = compute_pairwise_accuracy(label, predicted)
    recalls.append(recall)
    precisions.append(precision)
    F1s.append(F1)
    ACC.append(acc)
print("Training set:")
print("Average precision: ", np.mean(precisions))
print("Average recall: ", np.mean(recalls))
print("Average F1: ", np.mean(F1s))
print(f"Average ACC: {np.mean(ACC)}%")

