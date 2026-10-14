"""
AML Graph Neural Network Orchestrator & Forensic Engine
Handles graph normalization, training loops, multi-model benchmarking,
explainable AI (XAI) attention extraction, and Suspicious Activity Reports (SAR).
"""

import time
import numpy as np
import networkx as nx
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, average_precision_score
from sklearn.model_selection import train_test_split

from aml_generator import AMLGraphGenerator
from gnn_models import GCNClassifier, GATClassifier, GraphSAGEClassifier, BaselineMLP, FocalLoss

class AMLEngine:
    def __init__(self, seed=42):
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = AMLGraphGenerator(seed=seed)
        self.current_dataset = None
        self.models = {}
        self.active_model_name = "GAT"
        self.train_history = []
        self.split_indices = {}
        self.cached_metrics = {}

        # Initialize with standard benchmark dataset (260 accounts)
        self.load_benchmark_dataset(num_nodes=260, illicit_ratio=0.08)

    def load_benchmark_dataset(self, num_nodes=260, illicit_ratio=0.08):
        """Generates and preprocesses a new synthetic financial benchmark graph."""
        self.current_dataset = self.generator.generate_network(num_nodes=num_nodes, illicit_ratio=illicit_ratio)
        self._prepare_tensors()
        return self.get_graph_summary()

    def load_from_parsed_data(self, dataset):
        """
        Load an externally built dataset (e.g. from CSV upload) into the engine.

        The dataset dict must contain:
          graph    - nx.DiGraph with integer node IDs
          X        - np.ndarray (N, F) float32 feature matrix
          y        - np.ndarray (N,) int64 label vector (0=licit, 1=illicit)
          nodes    - list of node dicts (id, label, is_illicit, typology, …)
          edges    - list of edge dicts (from, to, amount, is_illicit, typology)
          num_illicit / num_licit / illicit_ratio

        Clears any previously trained models so stale weights are not applied
        to the new graph topology.
        """
        self.current_dataset = dataset
        self.models = {}           # reset trained models — topology changed
        self.train_history = []
        self.cached_metrics = {}
        self._prepare_tensors()
        return self.get_graph_summary()

    def _prepare_tensors(self):
        """Converts graph into PyTorch tensors and computes normalized adjacency matrices."""
        G = self.current_dataset["graph"]
        X = self.current_dataset["X"]
        y = self.current_dataset["y"]
        N = len(G.nodes())

        # Adjacency matrix with self-loops
        A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()), weight=None)
        A_loop = A + np.eye(N)

        # Degree matrix for GCN symmetric normalization: D^(-1/2) * A * D^(-1/2)
        deg = np.sum(A_loop, axis=1)
        deg_inv_sqrt = np.zeros_like(deg, dtype=float)
        np.power(deg, -0.5, where=deg > 0, out=deg_inv_sqrt)
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A_loop @ D_inv_sqrt

        self.tensors = {
            "X": torch.FloatTensor(X).to(self.device),
            "y": torch.LongTensor(y).to(self.device),
            "A_raw": torch.FloatTensor(A_loop).to(self.device),
            "A_norm": torch.FloatTensor(A_norm).to(self.device)
        }

        # Train / Val / Test split (resilient to class counts)
        indices = np.arange(N)
        unique_classes, counts = np.unique(y, return_counts=True)
        can_stratify = (len(unique_classes) >= 2) and (np.min(counts) >= 4) and (N >= 10)

        if can_stratify:
            try:
                train_idx, test_idx = train_test_split(indices, test_size=0.30, stratify=y, random_state=self.seed)
                test_unique, test_counts = np.unique(y[test_idx], return_counts=True)
                if len(test_unique) >= 2 and np.min(test_counts) >= 2:
                    val_idx, test_idx = train_test_split(test_idx, test_size=0.50, stratify=y[test_idx], random_state=self.seed)
                else:
                    val_idx, test_idx = train_test_split(test_idx, test_size=0.50, random_state=self.seed)
            except Exception:
                train_idx, test_idx = train_test_split(indices, test_size=0.30, random_state=self.seed)
                val_idx, test_idx = train_test_split(test_idx, test_size=0.50, random_state=self.seed)
        else:
            test_size = 0.30 if N >= 10 else 0.20
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=self.seed)
            val_idx, test_idx = train_test_split(test_idx, test_size=0.50, random_state=self.seed)

        self.split_indices = {
            "train": torch.LongTensor(train_idx).to(self.device),
            "val": torch.LongTensor(val_idx).to(self.device),
            "test": torch.LongTensor(test_idx).to(self.device)
        }

    def train_model(self, model_name="GAT", epochs=120, lr=0.015, weight_decay=5e-4, loss_type="focal"):
        """Trains the specified GNN or baseline architecture."""
        # Normalize model name aliases
        if model_name in ("Baseline MLP", "Baseline", "MLP"):
            model_name = "Baseline (MLP)"
        self.active_model_name = model_name
        in_dim = self.tensors["X"].shape[1]

        if model_name == "GCN":
            model = GCNClassifier(in_features=in_dim, hidden_dim=64, num_classes=2, dropout=0.25).to(self.device)
            adj_tensor = self.tensors["A_norm"]
        elif model_name == "GAT":
            model = GATClassifier(in_features=in_dim, hidden_dim=64, num_classes=2, dropout=0.25).to(self.device)
            adj_tensor = self.tensors["A_raw"]
        elif model_name == "GraphSAGE":
            model = GraphSAGEClassifier(in_features=in_dim, hidden_dim=64, num_classes=2, dropout=0.25).to(self.device)
            adj_tensor = self.tensors["A_raw"]
        elif model_name == "Baseline (MLP)":
            model = BaselineMLP(in_features=in_dim, hidden_dim=64, num_classes=2, dropout=0.25).to(self.device)
            adj_tensor = self.tensors["A_raw"]
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Handle class imbalance
        y_train = self.tensors["y"][self.split_indices["train"]].cpu().numpy()
        num_licit = np.sum(y_train == 0)
        num_illicit = max(1, np.sum(y_train == 1))
        pos_weight = float(num_licit / num_illicit)

        if loss_type == "focal":
            criterion = FocalLoss(alpha=0.75, gamma=2.0)
        else:
            weights = torch.FloatTensor([1.0, pos_weight]).to(self.device)
            criterion = torch.nn.CrossEntropyLoss(weight=weights)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

        X = self.tensors["X"]
        y = self.tensors["y"]
        train_idx = self.split_indices["train"]
        val_idx = self.split_indices["val"]

        history = []
        best_val_auc = -1.0
        best_state = None

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(X, adj_tensor)
            loss = criterion(logits[train_idx], y[train_idx])
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(X, adj_tensor)
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

            if epoch % 5 == 0 or epoch == epochs:
                history.append({
                    "epoch": epoch,
                    "train_loss": round(float(loss.item()), 4),
                    "val_auc": round(float(val_auc), 4),
                    "lr": round(optimizer.param_groups[0]['lr'], 6)
                })

        train_duration = time.time() - start_time

        # Restore best checkpoint
        if best_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self.models[model_name] = model
        self.train_history = history

        # Calculate final test metrics
        test_metrics = self.evaluate_model(model_name=model_name, threshold=0.50)
        test_metrics["train_time_sec"] = round(train_duration, 3)
        self.cached_metrics[model_name] = test_metrics

        return {
            "model_name": model_name,
            "history": history,
            "metrics": test_metrics,
            "train_time_sec": round(train_duration, 3)
        }

    def evaluate_model(self, model_name=None, threshold=0.50):
        """Computes comprehensive AML metrics and node-level risk predictions."""
        if model_name is None:
            model_name = self.active_model_name
        if model_name in ("Baseline MLP", "Baseline", "MLP"):
            model_name = "Baseline (MLP)"

        if model_name not in self.models:
            # Auto-train if not yet trained
            self.train_model(model_name=model_name, epochs=90)

        model = self.models[model_name]
        model.eval()

        adj_tensor = self.tensors["A_norm"] if model_name == "GCN" else self.tensors["A_raw"]
        with torch.no_grad():
            logits, embeddings = model(self.tensors["X"], adj_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        y_true = self.tensors["y"].cpu().numpy()
        test_idx = self.split_indices["test"].cpu().numpy()

        y_test = y_true[test_idx]
        probs_test = probs[test_idx]
        preds_test = (probs_test >= threshold).astype(int)

        try:
            auc = roc_auc_score(y_test, probs_test)
        except Exception:
            auc = 0.5

        try:
            pr_auc = average_precision_score(y_test, probs_test)
        except Exception:
            pr_auc = 0.0

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds_test, average='binary', zero_division=0
        )

        cm = confusion_matrix(y_test, preds_test, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Full network predictions
        all_preds = (probs >= threshold).astype(int)
        for i, node in enumerate(self.current_dataset["nodes"]):
            node["pred_risk_score"] = round(float(probs[i]), 4)
            node["pred_is_illicit"] = int(all_preds[i])

        return {
            "model_name": model_name,
            "decision_threshold": threshold,
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "total_test_samples": len(test_idx),
            "detected_illicit": int(tp + fp)
        }

    def benchmark_comparison(self):
        """Runs side-by-side benchmarking of GCN, GAT, GraphSAGE, and Baseline MLP."""
        architectures = ["GCN", "GAT", "GraphSAGE", "Baseline (MLP)"]
        results = []

        for arch in architectures:
            if arch not in self.models:
                self.train_model(model_name=arch, epochs=100)
            metrics = self.evaluate_model(model_name=arch, threshold=0.50)
            metrics["train_time_sec"] = self.cached_metrics.get(arch, {}).get("train_time_sec", 1.2)
            results.append(metrics)

        return results

    def explain_node(self, node_id):
        """
        Explainable AI (XAI) extraction for a specific account.
        Returns the k-hop ego network, topological features,
        and GAT attention weights showing transaction flow impact.
        """
        node_id = int(node_id)
        G = self.current_dataset["graph"]

        if node_id not in G:
            return {"error": f"Node {node_id} not found in transaction network."}

        # 1-hop and 2-hop ego network
        subgraph_nodes = set(nx.ego_graph(G, node_id, radius=2, undirected=True).nodes())
        subgraph = G.subgraph(subgraph_nodes)

        # Extract neighbor influence weights based on active architecture
        neighbor_influences = {}
        if self.active_model_name == "GAT" and "GAT" in self.models:
            attn_matrix = self.models["GAT"].get_attention_matrix()
            if attn_matrix is not None:
                for neighbor in G.neighbors(node_id):
                    weight = float(attn_matrix[node_id, neighbor])
                    neighbor_influences[neighbor] = round(weight, 4)
        elif self.active_model_name == "GCN":
            # Normalized Laplacian influence
            deg_i = max(1, G.in_degree(node_id) + G.out_degree(node_id))
            for neighbor in set(list(G.predecessors(node_id)) + list(G.successors(node_id))):
                deg_j = max(1, G.in_degree(neighbor) + G.out_degree(neighbor))
                weight = 1.0 / np.sqrt(float(deg_i * deg_j))
                neighbor_influences[neighbor] = round(float(weight), 4)
        elif self.active_model_name == "GraphSAGE":
            # Inductive sample aggregation weight
            total_neighbors = max(1, len(set(list(G.predecessors(node_id)) + list(G.successors(node_id)))))
            for neighbor in set(list(G.predecessors(node_id)) + list(G.successors(node_id))):
                weight = 1.0 / total_neighbors
                neighbor_influences[neighbor] = round(float(weight), 4)
        else:
            # Baseline MLP or fallback: feature similarity
            X = self.current_dataset["X"]
            x_i = X[node_id]
            norm_i = np.linalg.norm(x_i) + 1e-9
            for neighbor in set(list(G.predecessors(node_id)) + list(G.successors(node_id))):
                x_j = X[neighbor]
                sim = float(np.dot(x_i, x_j) / (norm_i * (np.linalg.norm(x_j) + 1e-9)))
                sim_norm = max(0.05, min(0.99, (sim + 1.0) / 2.0))
                neighbor_influences[neighbor] = round(sim_norm, 4)

        # Normalize weights so max is scaled cleanly
        if neighbor_influences:
            max_w = max(neighbor_influences.values()) or 1.0
            gat_attentions = {k: round(v / max_w, 4) for k, v in neighbor_influences.items()}
        else:
            gat_attentions = {}

        # Build ego-network payload — use .get() with defaults for CSV-uploaded graphs
        ego_nodes = []
        for n in sorted(subgraph.nodes()):
            n_meta = next((nd for nd in self.current_dataset["nodes"] if nd["id"] == n), {})
            ego_nodes.append({
                "id": n,
                "label": G.nodes[n].get('account_id', str(n)),
                "entity_type": G.nodes[n].get('entity_name', G.nodes[n].get('entity_type', 'Custom Entity')),
                "is_target": (n == node_id),
                "is_illicit": G.nodes[n].get('is_illicit', n_meta.get('is_illicit', 0)),
                "typology": G.nodes[n].get('typology', n_meta.get('typology', 'Unknown')),
                "risk_score": G.nodes[n].get('risk_score', n_meta.get('risk_score', 0.1)),
                "attention_weight": gat_attentions.get(n, None)
            })

        ego_edges = []
        for u, v, data in subgraph.edges(data=True):
            attn = None
            if u == node_id and v in gat_attentions:
                attn = gat_attentions[v]
            elif v == node_id and u in gat_attentions:
                attn = gat_attentions[u]

            ego_edges.append({
                "from": u,
                "to": v,
                "amount": data.get('amount', 0.0),
                "timestamp": data.get('timestamp', 1),
                "date": data.get('date', '—'),
                "time": data.get('time', '—'),
                "payment_type": data.get('pay_type', data.get('payment_type', 'Unknown')),
                "currency": data.get('currency', 'USD'),
                "sender_loc": data.get('sender_loc', 'Unknown'),
                "receiver_loc": data.get('receiver_loc', 'Unknown'),
                "typology": data.get('typology', 'Unknown'),
                "is_illicit": data.get('is_illicit', 0),
                "attention": attn
            })

        # Feature factors explanation
        target_meta_list = [nd for nd in self.current_dataset["nodes"] if nd["id"] == node_id]
        target_meta = target_meta_list[0] if target_meta_list else {}
        risk_factors = []

        if target_meta.get("pred_risk_score", 0) > 0.60:
            if "Smurf" in target_meta.get("typology", "") or target_meta.get("in_degree", 0) >= 4:
                risk_factors.append("High fan-in structuring: Multiple sub-$10k transfers aggregating within narrow timestamp.")
            if "Circular" in target_meta.get("typology", ""):
                risk_factors.append("Circular flow loop: Fund round-tripping observed returning to originating entity cluster.")
            if "Layering" in target_meta.get("typology", ""):
                risk_factors.append("Multi-tier layering hop: Rapid pass-through with minimal retention duration.")
            if target_meta.get("out_degree", 0) >= 4:
                risk_factors.append("High fan-out scatter: Rapid dispersion to multiple beneficiary accounts.")
            if len(risk_factors) == 0:
                risk_factors.append(f"Elevated {self.active_model_name} neighborhood anomaly score and abnormal transaction velocity.")
        else:
            risk_factors.append("Normal transactional distribution consistent with licit retail/commercial activity.")

        # Full transaction ledger for this account
        transactions = []
        for u, v, data in G.edges(data=True):
            if u == node_id or v == node_id:
                direction = "OUTFLOW" if u == node_id else "INFLOW"
                counterparty_id = v if u == node_id else u
                counterparty_label = G.nodes[counterparty_id].get('account_id', str(counterparty_id)) if counterparty_id in G else str(counterparty_id)
                transactions.append({
                    "direction": direction,
                    "from_node": u,
                    "to_node": v,
                    "from_account": G.nodes[u].get('account_id', str(u)) if u in G else str(u),
                    "to_account": G.nodes[v].get('account_id', str(v)) if v in G else str(v),
                    "counterparty": counterparty_label,
                    "amount": round(float(data.get('amount', 0.0)), 2),
                    "currency": data.get('currency', 'USD'),
                    "payment_type": data.get('pay_type', data.get('payment_type', 'Unknown')),
                    "date": data.get('date', '—'),
                    "time": data.get('time', '—'),
                    "sender_loc": data.get('sender_loc', data.get('sender_location', '—')),
                    "receiver_loc": data.get('receiver_loc', data.get('receiver_location', '—')),
                    "typology": data.get('typology', '—'),
                    "is_illicit": data.get('is_illicit', 0),
                    "custom_features": data.get('features', {})
                })

        # Sort transactions by amount or timestamp
        transactions.sort(key=lambda t: t['amount'], reverse=True)

        return {
            "node_id": node_id,
            "account_id": G.nodes[node_id].get('account_id', str(node_id)),
            "entity_name": G.nodes[node_id].get('entity_name', G.nodes[node_id].get('entity_type', 'Custom Entity')),
            "typology": G.nodes[node_id].get('typology', target_meta.get('typology', 'Unknown')),
            "is_illicit": G.nodes[node_id].get('is_illicit', target_meta.get('is_illicit', 0)),
            "predicted_risk": target_meta.get("pred_risk_score", 0.0),
            "predicted_illicit": target_meta.get("pred_is_illicit", 0),
            "total_in": G.nodes[node_id].get('total_in', target_meta.get('total_in', 0.0)),
            "total_out": G.nodes[node_id].get('total_out', target_meta.get('total_out', 0.0)),
            "in_degree": G.nodes[node_id].get('in_degree', G.in_degree(node_id)),
            "out_degree": G.nodes[node_id].get('out_degree', G.out_degree(node_id)),
            "betweenness": G.nodes[node_id].get('betweenness', target_meta.get('betweenness', 0.0)),
            "pagerank": G.nodes[node_id].get('pagerank', target_meta.get('pagerank', 0.0)),
            "risk_factors": risk_factors,
            "ego_nodes": ego_nodes,
            "ego_edges": ego_edges,
            "gat_attentions": gat_attentions,
            "transactions": transactions,
            "active_model": self.active_model_name
        }

    def get_edge_details(self, u, v):
        """Returns the full metadata and CSV features for a specific directed edge."""
        u, v = int(u), int(v)
        G = self.current_dataset["graph"]
        if not G.has_edge(u, v):
            if G.has_edge(v, u):
                u, v = v, u
            else:
                return {"error": f"Transaction edge between {u} and {v} not found."}
        d = G.edges[u, v]
        return {
            "from_node": u,
            "to_node": v,
            "from_account": G.nodes[u].get('account_id', str(u)),
            "to_account": G.nodes[v].get('account_id', str(v)),
            "amount": round(float(d.get('amount', 0.0)), 2),
            "currency": d.get('currency', 'USD'),
            "payment_type": d.get('pay_type', d.get('payment_type', 'Unknown')),
            "date": d.get('date', '—'),
            "time": d.get('time', '—'),
            "sender_loc": d.get('sender_loc', d.get('sender_location', '—')),
            "receiver_loc": d.get('receiver_loc', d.get('receiver_location', '—')),
            "typology": d.get('typology', '—'),
            "is_illicit": d.get('is_illicit', 0),
            "custom_features": d.get('features', {})
        }

    def generate_sar_report(self, node_id):
        """Generates a FinCEN Suspicious Activity Report (SAR) dossier for export."""
        xai = self.explain_node(node_id)
        if "error" in xai:
            return xai

        is_flagged = xai["predicted_risk"] >= 0.50
        action = "IMMEDIATE ACCOUNT FREEZE & FIU ESCALATION" if is_flagged else "STANDARD MONITORING / CLEARED"

        sar = {
            "report_id": f"SAR-GNN-{int(time.time())}-{node_id}",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "jurisdiction": "FinCEN / Egmont Group Standard",
            "subject_account": {
                "internal_id": xai["node_id"],
                "account_number": xai["account_id"],
                "entity_classification": xai["entity_name"],
                "inflow_total": f"${xai['total_in']:,.2f}",
                "outflow_total": f"${xai['total_out']:,.2f}",
                "net_exposure": f"${abs(xai['total_in'] - xai['total_out']):,.2f}"
            },
            "gnn_assessment": {
                "evaluated_model": self.active_model_name,
                "illicit_probability": f"{xai['predicted_risk'] * 100:.2f}%",
                "ground_truth_label": "ILLICIT" if xai["is_illicit"] else "LICIT",
                "predicted_status": "FLAGGED SUSPICIOUS" if is_flagged else "BENIGN",
                "detected_typology": xai["typology"],
                "network_centrality_percentile": f"{xai['betweenness'] * 100:.2f}%"
            },
            "forensic_trail": {
                "in_degree_connections": xai["in_degree"],
                "out_degree_connections": xai["out_degree"],
                "subgraph_hop_distance": "2-Hop Ego Graph",
                "primary_typology_indicators": xai["risk_factors"]
            },
            "recommended_regulatory_action": action
        }
        return sar

    def get_graph_summary(self):
        """Returns overview statistics of the current financial network."""
        ds = self.current_dataset
        return {
            "num_nodes": len(ds["nodes"]),
            "num_edges": len(ds["edges"]),
            "num_illicit": ds["num_illicit"],
            "num_licit": ds["num_licit"],
            "illicit_ratio": ds["illicit_ratio"],
            "nodes": ds["nodes"],
            "edges": ds["edges"],
            "active_model": self.active_model_name
        }
