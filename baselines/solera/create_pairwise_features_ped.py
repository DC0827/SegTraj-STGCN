import pickle
import os
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils import *
from models_solera import *
from concurrent.futures import ProcessPoolExecutor


data_path = "../../data/Sim"  # students03   NBA  Hotel  ETH Sim




train_path = os.path.join(data_path, "examples_train_unnormalized.pkl")
print("train_path: ", train_path)
valid_path = os.path.join(data_path, "examples_valid_unnormalized.pkl")
print("valid_path: ", valid_path)
test_path = os.path.join(data_path, "examples_test_unnormalized.pkl")
print("test_path: ", test_path)



#load training, validation and test data
with open(train_path, 'rb') as f:
    examples_train = pickle.load(f)
with open(valid_path, 'rb') as f:
    examples_valid = pickle.load(f)
with open(test_path, 'rb') as f:
    examples_test = pickle.load(f)
    
    
#create ground
ground = build_ground(examples_train)



pairwise_features_train = []
pairwise_features_valid = []
pairwise_features_test = []

# for example in tqdm(examples_train, desc="Processing Train"):
#     _,_,pairwise_features = compute_sims(example, ground)
#     pairwise_features_train.append(pairwise_features)
    
# for example in tqdm(examples_valid, desc="Processing Valid"):
#     _,_,pairwise_features = compute_sims(example, ground)
#     pairwise_features_valid.append(pairwise_features)
    
# for example in tqdm(examples_test, desc="Processing Test"):
#     _,_,pairwise_features = compute_sims(example, ground)
#     pairwise_features_test.append(pairwise_features)

def compute_pairwise(example):
    _, _, pairwise_features = compute_sims(example, ground)
    return pairwise_features

# 注意：ground 是共享变量，需在多进程中可用

with ProcessPoolExecutor(max_workers=8) as executor:
    pairwise_features_train = list(tqdm(executor.map(compute_pairwise, examples_train), total=len(examples_train), desc="Train"))

with ProcessPoolExecutor(max_workers=8) as executor:
    pairwise_features_valid = list(tqdm(executor.map(compute_pairwise, examples_valid), total=len(examples_valid), desc="Valid"))

with ProcessPoolExecutor(max_workers=8) as executor:
    pairwise_features_test = list(tqdm(executor.map(compute_pairwise, examples_test), total=len(examples_test), desc="Test"))
    
    


with open(os.path.join(data_path, "pairwise_features_train.pkl"), 'wb') as f:
    pickle.dump(pairwise_features_train, f)

with open(os.path.join(data_path, "pairwise_features_valid.pkl"), 'wb') as f:
    pickle.dump(pairwise_features_valid, f)
    
with open(os.path.join(data_path, "pairwise_features_test.pkl"), 'wb') as f:
    pickle.dump(pairwise_features_test, f)







