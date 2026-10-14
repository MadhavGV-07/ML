import csv, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

p = r'C:\Users\Madhav\Downloads\elliptic_bitcoin_dataset'
t0 = time.time()

# 1. Load classes
classes = {}
with open(os.path.join(p, 'elliptic_txs_classes.csv'), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row[1] == '1':
            classes[row[0]] = 1
        elif row[1] == '2':
            classes[row[0]] = 0

print(f"Classes loaded in {time.time()-t0:.2f}s")
t1 = time.time()

# 2. Load features for labeled nodes
# Features file has txId, time_step, feat_0, ..., feat_164 (167 cols)
train_X, train_y = [], []
val_X, val_y = [], []
test_X, test_y = [], []

with open(os.path.join(p, 'elliptic_txs_features.csv'), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        tx_id = row[0]
        if tx_id in classes:
            step = int(row[1])
            feats = [float(x) for x in row[2:]]
            lbl = classes[tx_id]
            if 1 <= step <= 29:
                train_X.append(feats)
                train_y.append(lbl)
            elif 30 <= step <= 34:
                val_X.append(feats)
                val_y.append(lbl)
            elif 35 <= step <= 49:
                test_X.append(feats)
                test_y.append(lbl)

print(f"Features loaded in {time.time()-t1:.2f}s: Train={len(train_y)}, Val={len(val_y)}, Test={len(test_y)}")
