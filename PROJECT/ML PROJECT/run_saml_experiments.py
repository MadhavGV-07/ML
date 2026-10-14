import time
import numpy as np
import networkx as nx
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

from app import build_dataset_from_stream
from gnn_models import GCNClassifier, GATClassifier, GraphSAGEClassifier, BaselineMLP, FocalLoss

def prepare_tensors(G, X, y, device, seed):
    N = len(G.nodes())
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()), weight=None)
    A_loop = A + np.eye(N)

    deg = np.sum(A_loop, axis=1)
    deg_inv_sqrt = np.zeros_like(deg, dtype=float)
    np.power(deg, -0.5, where=deg > 0, out=deg_inv_sqrt)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_loop @ D_inv_sqrt

    indices = np.arange(N)
    train_idx, test_idx = train_test_split(indices, test_size=0.30, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.50, stratify=y[test_idx], random_state=seed)

    return {
        "X": torch.FloatTensor(X).to(device),
        "y": torch.LongTensor(y).to(device),
        "A_raw": torch.FloatTensor(A_loop).to(device),
        "A_norm": torch.FloatTensor(A_norm).to(device),
        "train_idx": torch.LongTensor(train_idx).to(device),
        "val_idx": torch.LongTensor(val_idx).to(device),
        "test_idx": torch.LongTensor(test_idx).to(device),
    }

def train_and_eval(model_name, tensors, device, epochs=100, lr=0.015, weight_decay=5e-4):
    in_dim = tensors["X"].shape[1]
    if model_name == "GCN":
        model = GCNClassifier(in_features=in_dim, hidden_dim=64, num_classes=2, dropout=0.25).to(device)
        adj = tensors["A_norm"]
    elif model_name == "GAT":
        model = GATClassifier(in_features=in_dim, hidden_dim=64, num_classes=2, dropout=0.25).to(device)
        adj = tensors["A_raw"]
    elif model_name == "GraphSAGE":
        model = GraphSAGEClassifier(in_features=in_dim, hidden_dim=64, num_classes=2, dropout=0.25).to(device)
        adj = tensors["A_raw"]
    elif model_name in ("Baseline MLP", "Baseline (MLP)"):
        model = BaselineMLP(in_features=in_dim, hidden_dim=64, num_classes=2, dropout=0.25).to(device)
        adj = tensors["A_raw"]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    X = tensors["X"]
    y = tensors["y"]
    train_idx = tensors["train_idx"]
    val_idx = tensors["val_idx"]
    test_idx = tensors["test_idx"]

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X, adj)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X, adj)
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            y_val = y[val_idx].cpu().numpy()
            try:
                val_auc = roc_auc_score(y_val, val_probs[val_idx.cpu().numpy()])
            except Exception:
                val_auc = 0.5
            scheduler.step(val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        test_logits, _ = model(X, adj)
        probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()

    t_idx = test_idx.cpu().numpy()
    y_test = y[test_idx].cpu().numpy()
    p_test = probs[t_idx]
    preds = (p_test >= 0.50).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, p_test)
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(y_test, p_test)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cm": cm.tolist()
    }

def run_all():
    print("Loading SAML-D dataset...")
    col_map = {
        'sender': 'Sender_account', 'receiver': 'Receiver_account',
        'amount': 'Amount', 'label': 'Is_laundering', 'typology': 'Laundering_type',
        'currency': 'Payment_currency', 'payment_type': 'Payment_type',
        'sender_location': 'Sender_bank_location', 'receiver_location': 'Receiver_bank_location',
        'date': 'Date', 'time': 'Time'
    }
    dataset, sample_info = build_dataset_from_stream('SAML-D.csv', col_map, max_nodes=2500, scan_limit=250000)
    G = dataset['graph']
    X_full = dataset['X']
    y = dataset['y']
    device = torch.device("cpu")

    print(f"Graph: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}, Illicit={sum(y)}, Licit={len(y)-sum(y)}")
    
    seeds = [42, 43, 44, 45, 46]
    models = ["GCN", "GAT", "GraphSAGE", "Baseline MLP"]

    # 1. Clean Configuration (13 features: 0 to 12)
    print("\n--- Running SAML-D Clean Configuration (13 features) ---")
    X_clean = X_full[:, :13]
    clean_results = {m: [] for m in models}

    for seed in seeds:
        print(f"  Seed {seed}...")
        tensors = prepare_tensors(G, X_clean, y, device, seed)
        for m in models:
            res = train_and_eval(m, tensors, device, epochs=100)
            clean_results[m].append(res)

    # 2. Legacy Configuration (15 features: 0 to 14)
    print("\n--- Running SAML-D Legacy Configuration (15 features) ---")
    X_legacy = X_full[:, :15]
    legacy_results = {m: [] for m in models}

    for seed in seeds:
        print(f"  Seed {seed}...")
        tensors = prepare_tensors(G, X_legacy, y, device, seed)
        for m in models:
            res = train_and_eval(m, tensors, device, epochs=100)
            legacy_results[m].append(res)

    import json
    with open("saml_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "setup": {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "illicit_nodes": int(sum(y)),
                "licit_nodes": int(len(y) - sum(y)),
                "test_size": len(tensors["test_idx"])
            },
            "clean": clean_results,
            "legacy": legacy_results
        }, f, indent=2)

    print("\nSaved saml_experiment_results.json successfully!")

if __name__ == "__main__":
    run_all()
