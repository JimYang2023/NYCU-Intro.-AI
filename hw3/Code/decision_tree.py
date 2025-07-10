import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x
    
class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y,0)
        self.progress.close()

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        feature , threshold = self._best_split(X,y)
        if feature is None or depth == self.max_depth or len(np.unique(y)) == 1 or self.progress.n == self.progress.total:       
            labels , counts = np.unique(y,return_counts=True)
            max_count , common_label = 0 , 0
            for label , count in zip(labels,counts):
                if count > max_count:
                    common_label = label
                    max_count = count
            return {"label": common_label}
        
        self.progress.update(1)
        left_x , left_y , right_x , right_y = self._split_data(X,y,feature,threshold)
        left = self._build_tree(left_x,left_y,depth+1)
        right = self._build_tree(right_x,right_y,depth+1)
        return {"feature": feature, "threshold": threshold, "left": left, "right": right}
        
    def predict(self, X: np.ndarray)->np.ndarray:
        # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree_node):
        # (TODO) Recursive function to traverse the decision tree
        if "label" in tree_node:
            return tree_node["label"]
        if x[tree_node["feature"]] <= tree_node["threshold"]:
            return self._predict_tree(x,tree_node["left"])
        else:
            return self._predict_tree(x,tree_node["right"])

    def _split_data(self,X: np.ndarray, y: np.ndarray, feature_index: int, threshold: float):
        # (TODO) split one node into left and right node
        feature_value = X[:,feature_index]
        left_mask = feature_value <= threshold
        right_mask = ~left_mask
        return  X[left_mask],y[left_mask],X[right_mask],y[right_mask]

    def _best_split(self,X: np.ndarray, y: np.ndarray):
        # (TODO) Use Information Gain to find the best split for a dataset
        best_gain = -1
        best_feature = None
        best_threshold = 0
        current_entropy = self._entropy(y)
        feature_number = X.shape[1]
        for idx in tqdm(range(feature_number),desc='best split',leave=False):
            values = X[:,idx]
            thresholds = np.unique(values)
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                p = float(np.sum(left_mask))/len(y)
                if p==0 or p==1:
                    continue
                gain = current_entropy - p * self._entropy(y[left_mask]) - (1-p)*self._entropy(y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = idx
                    best_threshold = threshold            
        return best_feature, best_threshold

    def _entropy(self,y: np.ndarray)->float:
        # (TODO) Return the entropy
        val = 0.0
        total = len(y)
        _ , counts = np.unique(y,return_counts=True)
        for count in counts:
            p = count/total
            val -= p*np.log2(p)
        return val

def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and labels
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for image , label in dataloader:
            image,label = image.to(device),label.to(device)
            feature = model(image)
            features.append(feature.cpu())
            labels.append(label.cpu())
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    return features, labels

def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and path of the images
    model.eval()
    features = []
    paths = []
    idx = 1
    with torch.no_grad():
        for images,path in dataloader:
            images = images.to(device)
            feature = model(images)
            features.append(feature.cpu())
            for i in path:
                paths.append(idx)
                idx += 1
    features = torch.cat(features).numpy()
    return features, paths