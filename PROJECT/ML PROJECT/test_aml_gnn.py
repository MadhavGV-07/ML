"""
Automated Unit and Integration Tests for AML GNN Platform
"""

import unittest
import numpy as np
import torch
from aml_generator import AMLGraphGenerator
from gnn_models import GCNClassifier, GATClassifier, GraphSAGEClassifier, BaselineMLP, FocalLoss
from engine import AMLEngine
from app import app

class TestAMLGNN(unittest.TestCase):
    def setUp(self):
        self.generator = AMLGraphGenerator(seed=42)

    def test_generator(self):
        ds = self.generator.generate_network(num_nodes=80, illicit_ratio=0.10)
        self.assertIn("graph", ds)
        self.assertEqual(len(ds["nodes"]), 80)
        self.assertGreater(len(ds["edges"]), 50)
        self.assertGreater(ds["num_illicit"], 0)
        self.assertEqual(ds["X"].shape[0], 80)
        self.assertEqual(ds["X"].shape[1], 15)
        self.assertEqual(len(ds["y"]), 80)
        print("[OK] AMLGraphGenerator test passed.")

    def test_gnn_architectures(self):
        N = 20
        F = 15
        x = torch.randn(N, F)
        adj = torch.eye(N) + torch.bernoulli(torch.full((N, N), 0.1))

        # 1. GCN
        gcn = GCNClassifier(in_features=F, hidden_dim=32, num_classes=2)
        out, _ = gcn(x, adj)
        self.assertEqual(out.shape, (N, 2))

        # 2. GAT
        gat = GATClassifier(in_features=F, hidden_dim=32, num_classes=2)
        out, _ = gat(x, adj)
        self.assertEqual(out.shape, (N, 2))
        self.assertIsNotNone(gat.get_attention_matrix())

        # 3. GraphSAGE
        sage = GraphSAGEClassifier(in_features=F, hidden_dim=32, num_classes=2)
        out, _ = sage(x, adj)
        self.assertEqual(out.shape, (N, 2))

        # 4. Baseline MLP
        mlp = BaselineMLP(in_features=F, hidden_dim=32, num_classes=2)
        out, _ = mlp(x)
        self.assertEqual(out.shape, (N, 2))

        # 5. Focal Loss
        fl = FocalLoss()
        targets = torch.randint(0, 2, (N,))
        loss = fl(out, targets)
        self.assertGreater(float(loss.item()), 0.0)
        print("[OK] All GNN Model forward passes and Focal Loss verified.")

    def test_engine_orchestration(self):
        engine = AMLEngine(seed=42)
        engine.load_benchmark_dataset(num_nodes=60, illicit_ratio=0.10)
        
        # Fast train GAT
        res = engine.train_model(model_name="GAT", epochs=15)
        self.assertIn("history", res)
        self.assertIn("metrics", res)
        
        # Test explainability XAI
        xai = engine.explain_node(0)
        self.assertIn("risk_factors", xai)
        self.assertIn("ego_nodes", xai)

        # Test SAR report
        sar = engine.generate_sar_report(0)
        self.assertIn("report_id", sar)
        self.assertIn("subject_account", sar)
        print("[OK] AMLEngine training, XAI and SAR report verified.")

    def test_flask_endpoints(self):
        client = app.test_client()
        
        # GET /
        resp = client.get('/')
        self.assertEqual(resp.status_code, 200)

        # GET /api/graph
        resp = client.get('/api/graph')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('nodes', data)

        # POST /api/predict
        resp = client.post('/api/predict', json={'threshold': 0.50})
        self.assertEqual(resp.status_code, 200)

        # GET /api/explain/0
        resp = client.get('/api/explain/0')
        self.assertEqual(resp.status_code, 200)
        print("[OK] Flask REST API endpoints verified.")

if __name__ == '__main__':
    unittest.main()
