import os
import numpy as np
import pickle
from sklearn.svm import SVC

from sknetwork.topology import get_connected_components
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils import *
from tqdm import tqdm

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def create_edgeNode_relation(num_nodes, self_loops=False):
    if self_loops:
        indices = np.ones([num_nodes, num_nodes])
    else:
        indices = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
    rel_rec = np.array(encode_onehot(np.where(indices)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(indices)[1]), dtype=np.float32)
    
    return rel_rec, rel_send 


path = "../../data/NBA"   #ETH  students03 NBA  Hotel  Sim


# Load training examples
with open(os.path.join(path, "examples_train_unnormalized.pkl"), 'rb') as f:
    examples_train = pickle.load(f)
# Load training velocities
with open(os.path.join(path, "tensors_train.pkl"), 'rb') as f:    
    velocities_train = pickle.load(f)
    velocities_train = [ 
    sample[:,:,2:4]
    for sample in velocities_train  # 遍历样本
    ]
# Load training labels
with open(os.path.join(path, "labels_train.pkl"), 'rb') as f:
    labels_train = pickle.load(f)
    
# Load test examples
with open(os.path.join(path, "examples_test_unnormalized.pkl"), 'rb') as f:
    examples_test = pickle.load(f)
# Load test velocities
with open(os.path.join(path, "tensors_test.pkl"), 'rb') as f:
    velocities_test = pickle.load(f)
    velocities_test = [
        sample[:,:,2:4]
        for sample in velocities_test 
    ]

# Load test labels
with open(os.path.join(path, "labels_test.pkl"), 'rb') as f:
    labels_test = pickle.load(f)

# Feature Engineering on Training Data
# Process training labels
labels_train = np.concatenate(labels_train)
labels_train[labels_train == 0] = -1
print("Labels shape: ", labels_train.shape)

# Distances Histogram
hist_dist_train = []
distance_train_max = -np.inf
distance_train_min = np.inf


for example in tqdm(examples_train, desc="Building distance histogram"):

    n_atoms = example.shape[0]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    example_re = example.reshape((example.shape[0], -1))
    senders = np.matmul(rel_send, example_re)
    receivers = np.matmul(rel_rec, example_re)

    senders = senders.reshape((senders.shape[0], example.shape[1], example.shape[2]))
    receivers = receivers.reshape((receivers.shape[0], example.shape[1], example.shape[2]))

    distance_example = senders - receivers
    distance_example = distance_example ** 2
    distance_example = distance_example.sum(-1)
    distance_example = np.sqrt(distance_example)

    distance_example_max = distance_example.max()
    distance_example_min = distance_example.min()
    
    if distance_train_max < distance_example_max:
        distance_train_max = distance_example_max
    if distance_train_min > distance_example_min:
        distance_train_min = distance_example_min

for example in tqdm(examples_train, desc="Building distance histograms"):

    n_atoms = example.shape[0]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    example_re = example.reshape((example.shape[0], -1))
    senders = np.matmul(rel_send, example_re)
    receivers = np.matmul(rel_rec, example_re)

    senders = senders.reshape((senders.shape[0], example.shape[1], example.shape[2]))
    receivers = receivers.reshape((receivers.shape[0], example.shape[1], example.shape[2]))

    distance_example = senders - receivers
    distance_example = distance_example ** 2
    distance_example = distance_example.sum(-1)
    distance_example = np.sqrt(distance_example)

    bins = np.arange(distance_train_min, distance_train_max, 4)

    hist_dist_example = []
    for d in distance_example:
        if np.histogram(d, bins=bins)[0].sum() > 0:
            hist_d = np.histogram(d, bins=bins)[0] / np.histogram(d, bins=bins)[0].sum()
        else:
            hist_d = np.histogram(d, bins=bins)[0]
        hist_dist_example.append(hist_d)

    hist_dist_train = hist_dist_train + hist_dist_example 
    
hist_dist_train = np.array(hist_dist_train)

# Normalized Histogram of Speed Difference
hist_speedDiff_train = []
speedDiff_train_max = -np.inf
speedDiff_train_min = np.inf


for velocity in tqdm(velocities_train, desc="Building speed difference histogram"):

    n_atoms = velocity.shape[0]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    velocity_re = velocity.reshape((velocity.shape[0], -1))
    senders = np.matmul(rel_send, velocity_re)
    receivers = np.matmul(rel_rec, velocity_re)

    senders = senders.reshape((senders.shape[0], velocity.shape[1], velocity.shape[2]))
    receivers = receivers.reshape((receivers.shape[0], velocity.shape[1], velocity.shape[2]))

    speed_senders = np.sqrt((senders ** 2).sum(-1))
    speed_receivers = np.sqrt((receivers ** 2).sum(-1))

    diff_speed = np.abs(speed_receivers - speed_senders)

    speedDiff_velocity_max = diff_speed.max()
    speedDiff_velocity_min = diff_speed.min()
    
    if speedDiff_train_max < speedDiff_velocity_max:
        speedDiff_train_max = speedDiff_velocity_max
    if speedDiff_train_min > speedDiff_velocity_min:
        speedDiff_train_min = speedDiff_velocity_min

for velocity in tqdm(velocities_train, desc="Building direction difference histogram"):

    n_atoms = velocity.shape[0]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    velocity_re = velocity.reshape((velocity.shape[0], -1))

    senders = np.matmul(rel_send, velocity_re)
    receivers = np.matmul(rel_rec, velocity_re)

    senders = senders.reshape((senders.shape[0], velocity.shape[1], velocity.shape[2]))
    receivers = receivers.reshape((receivers.shape[0], velocity.shape[1], velocity.shape[2]))

    speed_senders = np.sqrt((senders ** 2).sum(-1))
    speed_receivers = np.sqrt((receivers ** 2).sum(-1))

    diff_speed = np.abs(speed_receivers - speed_senders)

    bins = np.arange(speedDiff_train_min, speedDiff_train_max, 0.5)

    hist_diff_speed = []
    for edge in diff_speed:
        if np.histogram(edge, bins=bins)[0].sum() > 0:
            hist = np.histogram(edge, bins=bins)[0] / np.histogram(edge, bins=bins)[0].sum()
        else:
            hist = np.histogram(edge, bins=bins)[0]
        hist_diff_speed.append(hist)
    hist_speedDiff_train = hist_speedDiff_train + hist_diff_speed

hist_speedDiff_train = np.array(hist_speedDiff_train)

# Normalized Histogram of Absolute Difference in direction
hist_direcDiff_train = []
directDiff_train_max = np.pi + 0.25 * np.pi
directDiff_train_min = 0

for velocity in tqdm(velocities_train, desc="Building speed difference histogram"):
    n_atoms = velocity.shape[0]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    velocity_re = velocity.reshape((velocity.shape[0], -1))

    senders = np.matmul(rel_send, velocity_re)
    receivers = np.matmul(rel_rec, velocity_re)

    senders = senders.reshape((senders.shape[0], velocity.shape[1], velocity.shape[2]))
    receivers = receivers.reshape((receivers.shape[0], velocity.shape[1], velocity.shape[2]))

    tans_senders = senders[:, :, 1] / senders[:, :, 0]
    direc_senders = np.arctan(tans_senders)
    tans_receivers = receivers[:, :, 1] / receivers[:, :, 0]
    direc_receivers = np.arctan(tans_receivers)

    diff_direc = np.abs(direc_senders - direc_receivers)

    bins = np.arange(directDiff_train_min, directDiff_train_max, 0.25 * np.pi)

    hist_diff_direc = []
    for edge in diff_direc:
        if np.histogram(edge, bins=bins)[0].sum() > 0:
            hist = np.histogram(edge, bins=bins)[0] / np.histogram(edge, bins=bins)[0].sum()
        else:
            hist = np.histogram(edge, bins=bins)[0]
        hist_diff_direc.append(hist)
    hist_direcDiff_train = hist_direcDiff_train + hist_diff_direc

hist_direcDiff_train = np.array(hist_direcDiff_train)

# Normalized Histogram of Absolute Difference in velocity direction and relative position
hist_diffPV_train = []

for i in tqdm(range(len(examples_train)), desc="Building velocity-position difference histogram"):

    location = examples_train[i]
    velocity = velocities_train[i]

    n_atoms = location.shape[0]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    location_re = location.reshape((location.shape[0], -1))
    velocity_re = velocity.reshape((velocity.shape[0], -1))

    senders_vel = np.matmul(rel_send, velocity_re)
    receivers_vel = np.matmul(rel_rec, velocity_re)
    senders_loc = np.matmul(rel_send, location_re)
    receivers_loc = np.matmul(rel_rec, location_re)

    senders_vel = senders_vel.reshape(senders_vel.shape[0], velocity.shape[1], velocity.shape[2])
    receivers_vel = receivers_vel.reshape(receivers_vel.shape[0], velocity.shape[1], velocity.shape[2])
    senders_loc = senders_loc.reshape(senders_loc.shape[0], location.shape[1], location.shape[2])
    receivers_loc = receivers_loc.reshape(receivers_loc.shape[0], location.shape[1], location.shape[2])

    epsilon = 1e-6  # 一个非常小的数，用于避免除以零

    diff_locations = senders_loc - receivers_loc
    diff_locations_tans = diff_locations[:, :, 1] / (diff_locations[:, :, 0] + epsilon)
    relative_positions = np.arctan(diff_locations_tans)

    diff_velocities = senders_vel - receivers_vel
    diff_velocities_tans = diff_velocities[:, :, 1] / (diff_velocities[:, :, 0] + epsilon)
    relative_vel = np.arctan(diff_velocities_tans)

    diff_vel_loc = np.abs(relative_positions - relative_vel)

    bins = np.arange(0, np.pi + 0.25 * np.pi, 0.25 * np.pi)

    hist_diffPV = []
    for edge in diff_vel_loc:
        if np.histogram(edge, bins=bins)[0].sum() > 0:
            hist = np.histogram(edge, bins=bins)[0] / np.histogram(edge, bins=bins)[0].sum()
        else:
            hist = np.histogram(edge, bins=bins)[0]
        hist_diffPV.append(hist)

    hist_diffPV_train = hist_diffPV_train + hist_diffPV

hist_diffPV_train = np.array(hist_diffPV_train)

# Building Features
hist_feat = np.concatenate([hist_dist_train, hist_speedDiff_train, hist_direcDiff_train, hist_diffPV_train], axis=-1)
hist_feat.shape

# Train model on Training Dataset
# 训练模型
clf = SVC(gamma="auto")
clf.fit(hist_feat, labels_train)

# Test Model on Test Dataset
recall_test = []
precision_test = []
F1_test = []
predicted_edges_all = []
ACC_test = []

for i in tqdm(range(len(examples_test)), desc="Testing Model on Test Dataset"):  
    example = examples_test[i]
    velocity = velocities_test[i]
    label = labels_test[i]
    n_atoms = example.shape[0]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)

    example_re = example.reshape((example.shape[0], -1))
    senders_loc = np.matmul(rel_send, example_re)
    receivers_loc = np.matmul(rel_rec, example_re)

    senders_loc = senders_loc.reshape((senders_loc.shape[0], example.shape[1], example.shape[2]))
    receivers_loc = receivers_loc.reshape((receivers_loc.shape[0], example.shape[1], example.shape[2]))

    velocity_re = velocity.reshape((velocity.shape[0], -1))
    senders_vel = np.matmul(rel_send, velocity_re)
    receivers_vel = np.matmul(rel_rec, velocity_re)

    senders_vel = senders_vel.reshape((senders_vel.shape[0], velocity.shape[1], velocity.shape[2]))
    receivers_vel = receivers_vel.reshape((receivers_vel.shape[0], velocity.shape[1], velocity.shape[2]))

    distance_example = senders_loc - receivers_loc
    distance_example = distance_example ** 2
    distance_example = distance_example.sum(-1)
    distance_example = np.sqrt(distance_example)

    bins = np.arange(distance_train_min, distance_train_max, 4)

    hist_dist_example = []
    for d in distance_example:
        if np.histogram(d, bins=bins)[0].sum() > 0:
            hist_d = np.histogram(d, bins=bins)[0] / np.histogram(d, bins=bins)[0].sum()
        else:
            hist_d = np.histogram(d, bins=bins)[0]
        hist_dist_example.append(hist_d)

    hist_dist_example = np.array(hist_dist_example)

    speed_senders = np.sqrt((senders_vel ** 2).sum(-1))
    speed_receivers = np.sqrt((receivers_vel ** 2).sum(-1))

    diff_speed = np.abs(speed_receivers - speed_senders)

    bins = np.arange(speedDiff_train_min, speedDiff_train_max, 0.5)

    hist_diffSpeed_example = []
    for edge in diff_speed:
        if np.histogram(edge, bins=bins)[0].sum() > 0:
            hist = np.histogram(edge, bins=bins)[0] / np.histogram(edge, bins=bins)[0].sum()
        else:
            hist = np.histogram(edge, bins=bins)[0]
        hist_diffSpeed_example.append(hist)

    hist_diffSpeed_example = np.array(hist_diffSpeed_example)

    tans_senders_vel = senders_vel[:, :, 1] / senders_vel[:, :, 0]
    direc_senders_vel = np.arctan(tans_senders_vel)
    tans_receivers_vel = receivers_vel[:, :, 1] / receivers_vel[:, :, 0]
    direc_receivers_vel = np.arctan(tans_receivers_vel)

    diff_direcVel = np.abs(direc_senders_vel - direc_receivers_vel)

    bins = np.arange(0., np.pi + 0.25 * np.pi, 0.25 * np.pi)

    hist_diffVel_example = []
    for edge in diff_direcVel:
        if np.histogram(edge, bins=bins)[0].sum() > 0:
            hist = np.histogram(edge, bins=bins)[0] / np.histogram(edge, bins=bins)[0].sum()
        else:
            hist = np.histogram(edge, bins=bins)[0]
        hist_diffVel_example.append(hist)

    hist_diffVel_example = np.array(hist_diffVel_example)

    diff_locations = senders_loc - receivers_loc
    diff_locations_tans = diff_locations[:, :, 1] / (diff_locations[:, :, 0] + epsilon)
    relative_positions = np.arctan(diff_locations_tans)

    diff_velocities = senders_vel - receivers_vel
    diff_velocities_tans = diff_velocities[:, :, 1] / (diff_velocities[:, :, 0] + + epsilon)
    relative_vel = np.arctan(diff_velocities_tans)

    diff_vel_loc = np.abs(relative_positions - relative_vel)

    bins = np.arange(0, np.pi + 0.25 * np.pi, 0.25 * np.pi)

    hist_diffVP_example = []
    for edge in diff_vel_loc:
        if np.histogram(edge, bins=bins)[0].sum() > 0:
            hist = np.histogram(edge, bins=bins)[0] / np.histogram(edge, bins=bins)[0].sum()
        else:
            hist = np.histogram(edge, bins=bins)[0]
        hist_diffVP_example.append(hist)

    hist_diffVP_example = np.array(hist_diffVP_example)

    hist_feat_example = np.concatenate([hist_dist_example, hist_diffSpeed_example, 
                                        hist_diffVel_example, hist_diffVP_example], axis=-1)

    predicted_edges = clf.predict(hist_feat_example)
    predicted_edges[predicted_edges == -1] = 0
    predicted_edges_all.append(predicted_edges)
    predicted_edges_diag = np.diag(predicted_edges)
    predicted_relation = np.matmul(rel_send.T, np.matmul(predicted_edges_diag, rel_rec))
    if predicted_relation.sum() == 0:
        pred_con = np.arange(n_atoms)
    else:
        pred_con = get_connected_components(predicted_relation)
        
    label_edges_diag = np.diag(label)
    label_relation = np.matmul(rel_send.T, np.matmul(label_edges_diag, rel_rec))
    if label_relation.sum() == 0:
        label_con = np.arange(n_atoms)
    else:
        label_con = get_connected_components(label_relation)
      
    recall, precision, F1 = compute_groupMitre_labels(label_con, pred_con)
    acc = compute_pairwise_accuracy_labels(label_con, pred_con)
    recall_test.append(recall)
    precision_test.append(precision)
    F1_test.append(F1)
    ACC_test.append(acc)

print("Average precision: ", np.mean(precision_test))
print("Average recall: ", np.mean(recall_test))
print("Average F1: ", np.mean(F1_test))
print(f"Average ACC: {np.mean(ACC_test)}%")

# Compute edge recall
def compute_confusion(preds, target):
    true_positive = ((preds[target == 1] == 1).sum())
    false_negative = ((preds[target == 1] == 0).sum())
    true_negative = ((preds[target == 0] == 0).sum())
    false_positive = ((preds[target == 0] == 1).sum())

    return true_negative, false_negative, false_positive, true_positive

preds = np.concatenate(predicted_edges_all)
labels_test_all = np.concatenate(labels_test)

tn, fn, fp, tp = compute_confusion(preds, labels_test_all)
print("True negative: ", tn)
print("False negative: ", fn)
print("False positive: ", fp)
print("True positive: ", tp)

print("Normalized tn: ", tn / (tn + fp))
print("Normalized fn: ", fn / (tp + fn))
print("Normalized fp: ", fp / (fp + tn))
print("Normalized tp: ", tp / (fn + tp))
