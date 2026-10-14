import csv, os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from gnn_models import GCNClassifier, GATClassifier, GraphSAGEClassifier, BaselineMLP, FocalLoss

p = r'C:\Users\Madhav\Downloads\elliptic_bitcoin_dataset'

def load_data():
    t0 = time.time()
    # 1. Classes
    classes = {}
    with open(os.path.join(p, 'elliptic_txs_classes.csv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1] == '1':
                classes[row[0]] = 1
            elif row[1] == '2':
                classes[row[0]] = 0

    # 2. Features by timestep
    timestep_nodes = {t: [] for t in range(1, 50)}
    node_features = {}
    with open(os.path.join(p, 'elliptic_txs_features.csv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            tx_id = row[0]
            step = int(row[1])
            timestep_nodes[step].append(tx_id)
            node_features[tx_id] = [float(x) for x in row[2:]]

    # 3. Edges by timestep
    tx_to_step = {}
    for step, nodes in timestep_nodes.items():
        for n in nodes:
            tx_to_step[n] = step

    step_edges = {t: [] for t in range(1, 50)}
    with open(os.path.join(p, 'elliptic_txs_edgelist.csv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            u, v = row[0], row[1]
            if u in tx_to_step and v in tx_to_step and tx_to_step[u] == tx_to_step[v]:
                step_edges[tx_to_step[u]].append((u, v))

    print(f"Data loaded in {time.time()-t0:.2f}s")
    return classes, timestep_nodes, node_features, step_edges

def prepare_step_tensors(step, timestep_nodes, node_features, step_edges, classes, num_features=165):
    nodes = timestep_nodes[step]
    N = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    # Feature matrix
    X_arr = np.array([node_features[n][:num_features] for n in nodes], dtype=np.float32)
    
    # Adjacency with self-loops
    adj = np.eye(N, dtype=np.float32)
    for u, v in step_edges[step]:
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i, j] = 1.0

    # Normalized adjacency for GCN
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.zeros_like(deg, dtype=float)
    np.power(deg, -0.5, where=deg > 0, out=deg_inv_sqrt)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    adj_norm = (D_inv_sqrt @ adj @ D_inv_sqrt).astype(np.float32)

    # Labels and mask
    labeled_idx = []
    labels = []
    for i, n in enumerate(nodes):
        if n in classes:
            labeled_idx.append(i)
            labels.append(classes[n])

    return {
        "X": torch.from_numpy(X_arr),
        "adj": torch.from_numpy(adj),
        "adj_norm": torch.from_numpy(adj_norm),
        "labeled_idx": torch.LongTensor(labeled_idx),
        "labels": torch.LongTensor(labels)
    }

def train_and_eval_elliptic(num_features=165, epochs=30):
    classes, timestep_nodes, node_features, step_edges = load_data()
    device = torch.device("cpu")

    # Precompute step tensors
    print(f"Building tensors for {num_features} features...")
    step_data = {}
    for s in range(1, 50):
        step_data[s] = prepare_step_tensors(s, timestep_nodes, node_features, step_edges, classes, num_features)

    train_steps = list(range(1, 30))
    val_steps = list(range(30, 35))
    test_steps = list(range(35, 50))

    models = ["GCN", "GAT", "GraphSAGE", "Baseline MLP"]
    results = {}

    for m_name in models:
        print(f"\n--- Training {m_name} (features={num_features}) ---")
        if m_name == "GCN":
            model = GCNClassifier(num_features, 64, 2, dropout=0.25).to(device)
        elif m_name == "GAT":
            model = GATClassifier(num_features, 64, 2, dropout=0.25).to(device)
        elif m_name == "GraphSAGE":
            model = GraphSAGEClassifier(num_features, 64, 2, dropout=0.25).to(device)
        elif m_name == "Baseline MLP":
            model = BaselineMLP(num_features, 64, 2, dropout=0.25).to(device)

        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_val_pr = -1.0
        best_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            for s in train_steps:
                d = step_data[s]
                if len(d["labeled_idx"]) == 0:
                    continue
                optimizer.zero_grad()
                adj_in = d["adj_norm"] if m_name == "GCN" else d["adj"]
                out, _ = model(d["X"], adj_in)
                loss = criterion(out[d["labeled_idx"]], d["labels"])
                loss.backward()
                optimizer.step()

            # Validation PR-AUC across val steps
            model.eval()
            val_probs, val_true = [], []
            with torch.no_grad():
                for s in val_steps:
                    d = step_data[s]
                    if len(d["labeled_idx"]) == 0:
                        continue
                    adj_in = d["adj_norm"] if m_name == "GCN" else d["adj"]
                    out, _ = model(d["X"], adj_in)
                    p = torch.softmax(out[d["labeled_idx"]], dim=1)[:, 1].numpy()
                    val_probs.extend(p)
                    val_true.extend(d["labels"].numpy())

            val_true = np.array(val_true)
            val_probs = np.array(val_probs)
            try:
                val_pr = average_precision_score(val_true, val_probs)
            except Exception:
                val_pr = 0.0

            if val_pr > best_val_pr:
                best_val_pr = val_pr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        # Evaluate on test steps (35 to 49)
        model.eval()
        test_probs, test_true = [], []
        with torch.no_grad():
            for s in test_steps:
                d = step_data[s]
                if len(d["labeled_idx"]) == 0:
                    continue
                adj_in = d["adj_norm"] if m_name == "GCN" else d["adj"]
                out, _ = model(d["X"], adj_in)
                p = torch.softmax(out[d["labeled_idx"]], dim=1)[:, 1].numpy()
                test_probs.extend(p)
                test_true.extend(d["labels"].numpy())

        test_true = np.array(test_true)
        test_probs = np.array(test_probs)
        preds = (test_probs >= 0.50).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(test_true, preds, average='binary', zero_division=0)
        try:
            roc_auc = roc_auc_score(test_true, test_probs)
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = average_precision_score(test_true, test_probs)
        except Exception:
            pr_auc = 0.0

        results[m_name] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
        }
        print(f"  {m_name}: Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

    return results

if __name__ == "__main__":
    print("=== ELLIPTIC EXPERIMENTS: ALL FEATURES (165) ===")
    res_165 = train_and_eval_elliptic(num_features=165, epochs=30)
    print("\n=== ELLIPTIC EXPERIMENTS: LOCAL FEATURES ONLY (93) ===")
    res_93 = train_and_eval_elliptic(num_features=93, epochs=30)

    with open("elliptic_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump({"all_features_165": res_165, "local_features_93": res_93}, f, indent=2)
    print("\nSaved elliptic_experiment_results.json!")
