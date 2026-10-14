"""
Anti-Money Laundering (AML) Synthetic & Benchmark Graph Generator
Modeled after Elliptic Bitcoin Dataset and AMLSim specifications.
Generates realistic financial networks with topological AML typologies:
- Smurfing / Structuring
- Circular Flow (Round-Tripping)
- Layering / Scatter-Gather
- Mule Rings
"""

import random
import numpy as np
import networkx as nx

class AMLGraphGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_network(self, num_nodes=250, illicit_ratio=0.08, num_typologies=None):
        """
        Generates a realistic financial transaction network with licit background
        traffic and injected money laundering typologies.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        G = nx.DiGraph()

        # 1. Initialize Nodes (Accounts)
        # Entity types: 0=Retail/Individual, 1=Merchant/Business, 2=Crypto Exchange, 3=Offshore/Shell
        for i in range(num_nodes):
            rand_val = random.random()
            if rand_val < 0.70:
                entity_type = 0  # Individual
                entity_name = "Retail Account"
            elif rand_val < 0.90:
                entity_type = 1  # Merchant
                entity_name = "Commercial Entity"
            elif rand_val < 0.97:
                entity_type = 2  # Exchange
                entity_name = "Crypto Exchange"
            else:
                entity_type = 3  # High-risk / Offshore
                entity_name = "Offshore Entity"

            G.add_node(i, 
                       id=i,
                       account_id=f"ACC-{1000 + i}",
                       entity_type=entity_type,
                       entity_name=entity_name,
                       is_illicit=0,
                       typology="Licit",
                       risk_score=round(random.uniform(0.01, 0.20), 4),
                       tenure_days=random.randint(30, 2000))

        # 2. Generate Licit Background Transactions (Scale-free network)
        # Most transactions are normal payments, invoices, retail purchases
        bg_edges = int(num_nodes * 1.8)
        for _ in range(bg_edges):
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u != v:
                # Normal transaction amount follows log-normal / Pareto distribution
                amount = round(float(np.random.lognormal(mean=5.5, sigma=1.2)), 2)
                amount = max(10.0, min(amount, 8500.0))  # mostly normal amounts
                timestamp = random.randint(1, 100)
                G.add_edge(u, v, 
                           amount=amount, 
                           timestamp=timestamp, 
                           currency="USD", 
                           is_illicit=0,
                           typology="Normal")

        # 3. Inject Money Laundering Typologies
        target_illicit_nodes = int(num_nodes * illicit_ratio)
        current_illicit_count = 0
        used_nodes = set()

        def get_fresh_nodes(count):
            available = [n for n in range(num_nodes) if n not in used_nodes and G.nodes[n]['entity_type'] != 2]
            if len(available) < count:
                available = [n for n in range(num_nodes) if n not in used_nodes]
            selected = random.sample(available, min(count, len(available)))
            for n in selected:
                used_nodes.add(n)
            return selected

        # Typology A: Structuring / Smurfing (Splitting large sum below $10,000)
        # Originator -> 4-7 Smurf Accounts -> Aggregator / Beneficiary
        num_smurf_patterns = max(1, target_illicit_nodes // 8)
        for _ in range(num_smurf_patterns):
            nodes = get_fresh_nodes(random.randint(5, 7))
            if len(nodes) >= 5:
                originator = nodes[0]
                beneficiary = nodes[-1]
                smurfs = nodes[1:-1]

                total_launder_fund = round(random.uniform(50000, 150000), 2)
                split_amount = round(total_launder_fund / len(smurfs), 2)
                # Keep individual amounts just below $10,000 threshold (e.g., $9,200 - $9,800)
                if split_amount > 9900:
                    split_amount = round(random.uniform(9200, 9800), 2)

                G.nodes[originator]['is_illicit'] = 1
                G.nodes[originator]['typology'] = "Structuring / Smurfing"
                G.nodes[originator]['risk_score'] = round(random.uniform(0.75, 0.98), 4)

                G.nodes[beneficiary]['is_illicit'] = 1
                G.nodes[beneficiary]['typology'] = "Structuring / Smurfing"
                G.nodes[beneficiary]['risk_score'] = round(random.uniform(0.80, 0.99), 4)

                ts_base = random.randint(10, 60)
                for s in smurfs:
                    G.nodes[s]['is_illicit'] = 1
                    G.nodes[s]['typology'] = "Mule (Smurfing)"
                    G.nodes[s]['risk_score'] = round(random.uniform(0.65, 0.90), 4)

                    # Origin -> Smurf
                    G.add_edge(originator, s, 
                               amount=split_amount, 
                               timestamp=ts_base + random.randint(1, 3), 
                               currency="USD", 
                               is_illicit=1, 
                               typology="Structuring Split")
                    # Smurf -> Beneficiary (minus small transaction fee)
                    mule_cut = round(split_amount * random.uniform(0.96, 0.99), 2)
                    G.add_edge(s, beneficiary, 
                               amount=mule_cut, 
                               timestamp=ts_base + random.randint(4, 7), 
                               currency="USD", 
                               is_illicit=1, 
                               typology="Structuring Aggregate")
                current_illicit_count += len(nodes)

        # Typology B: Circular Flow / Round-Tripping (Cycle to disguise origin)
        # v1 -> v2 -> v3 -> v4 -> v1
        num_cycles = max(1, target_illicit_nodes // 10)
        for _ in range(num_cycles):
            cycle_len = random.choice([3, 4, 5])
            nodes = get_fresh_nodes(cycle_len)
            if len(nodes) == cycle_len:
                cycle_fund = round(random.uniform(40000, 200000), 2)
                ts = random.randint(20, 70)
                for idx in range(cycle_len):
                    u = nodes[idx]
                    v = nodes[(idx + 1) % cycle_len]
                    G.nodes[u]['is_illicit'] = 1
                    G.nodes[u]['typology'] = "Circular Flow"
                    G.nodes[u]['risk_score'] = round(random.uniform(0.70, 0.95), 4)

                    # slight decay representing service fees / laundering commission
                    cycle_fund = round(cycle_fund * random.uniform(0.97, 0.99), 2)
                    G.add_edge(u, v, 
                               amount=cycle_fund, 
                               timestamp=ts + idx * 2, 
                               currency="USD", 
                               is_illicit=1, 
                               typology="Round-Tripping Cycle")
                current_illicit_count += cycle_len

        # Typology C: Layering / Scatter-Gather
        # Source -> Layer 1 mules -> Layer 2 mules -> Exit / Crypto Mixer
        num_layering = max(1, target_illicit_nodes // 12)
        for _ in range(num_layering):
            nodes = get_fresh_nodes(6)
            if len(nodes) >= 6:
                src = nodes[0]
                l1_a, l1_b = nodes[1], nodes[2]
                l2_a, l2_b = nodes[3], nodes[4]
                sink = nodes[5]

                for n in nodes:
                    G.nodes[n]['is_illicit'] = 1
                    G.nodes[n]['typology'] = "Layering / Scatter-Gather"
                    G.nodes[n]['risk_score'] = round(random.uniform(0.78, 0.98), 4)

                ts = random.randint(15, 50)
                initial_sum = round(random.uniform(80000, 300000), 2)
                half_sum = round(initial_sum / 2, 2)

                # Source fans out to L1
                G.add_edge(src, l1_a, amount=half_sum, timestamp=ts, currency="USD", is_illicit=1, typology="Layering Fan-Out")
                G.add_edge(src, l1_b, amount=half_sum, timestamp=ts + 1, currency="USD", is_illicit=1, typology="Layering Fan-Out")

                # L1 shuffles to L2
                G.add_edge(l1_a, l2_a, amount=round(half_sum * 0.98, 2), timestamp=ts + 3, currency="USD", is_illicit=1, typology="Layering Hop")
                G.add_edge(l1_b, l2_b, amount=round(half_sum * 0.98, 2), timestamp=ts + 4, currency="USD", is_illicit=1, typology="Layering Hop")

                # L2 gathers to Sink
                G.add_edge(l2_a, sink, amount=round(half_sum * 0.96, 2), timestamp=ts + 6, currency="USD", is_illicit=1, typology="Layering Fan-In")
                G.add_edge(l2_b, sink, amount=round(half_sum * 0.96, 2), timestamp=ts + 7, currency="USD", is_illicit=1, typology="Layering Fan-In")
                current_illicit_count += len(nodes)

        # 4. Extract Multi-Dimensional Features for Every Node
        # (Topological graph features + transaction volume statistics)
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        betweenness = nx.betweenness_centrality(G)
        pagerank = nx.pagerank(G, alpha=0.85)

        features = []
        labels = []
        node_metadata = []

        for node_id in sorted(G.nodes()):
            in_edges = list(G.in_edges(node_id, data=True))
            out_edges = list(G.out_edges(node_id, data=True))

            in_amounts = [d['amount'] for _, _, d in in_edges]
            out_amounts = [d['amount'] for _, _, d in out_edges]

            total_in = sum(in_amounts) if in_amounts else 0.0
            total_out = sum(out_amounts) if out_amounts else 0.0
            net_flow = total_in - total_out
            in_out_ratio = (total_in + 1.0) / (total_out + 1.0)

            avg_in = np.mean(in_amounts) if in_amounts else 0.0
            avg_out = np.mean(out_amounts) if out_amounts else 0.0
            max_tx = max(in_amounts + out_amounts) if (in_amounts or out_amounts) else 0.0

            # Rapid flow velocity: transactions per distinct timestamp
            all_ts = [d['timestamp'] for _, _, d in in_edges + out_edges]
            velocity = len(all_ts) / (len(set(all_ts)) + 1e-5) if all_ts else 0.0

            k_in = in_degree.get(node_id, 0)
            k_out = out_degree.get(node_id, 0)
            deg_centrality = (k_in + k_out) / (max(1, 2 * (num_nodes - 1)))
            betw = betweenness.get(node_id, 0.0)
            pr = pagerank.get(node_id, 0.0)

            entity_type = G.nodes[node_id]['entity_type']
            tenure = G.nodes[node_id]['tenure_days'] / 2000.0  # normalized

            # Feature vector: 12 engineered features
            feat = [
                np.log1p(total_in),
                np.log1p(total_out),
                np.log1p(net_flow if net_flow > 0 else 0),
                np.log1p(abs(net_flow) if net_flow < 0 else 0),
                min(in_out_ratio, 50.0) / 50.0,
                np.log1p(avg_in),
                np.log1p(avg_out),
                np.log1p(max_tx),
                velocity,
                k_in / 20.0,
                k_out / 20.0,
                betw * 10.0,
                pr * 100.0,
                entity_type / 3.0,
                tenure
            ]
            features.append(feat)

            label = G.nodes[node_id]['is_illicit']
            labels.append(label)

            # Store computed attributes on node
            G.nodes[node_id]['total_in'] = round(total_in, 2)
            G.nodes[node_id]['total_out'] = round(total_out, 2)
            G.nodes[node_id]['in_degree'] = k_in
            G.nodes[node_id]['out_degree'] = k_out
            G.nodes[node_id]['max_tx'] = round(max_tx, 2)
            G.nodes[node_id]['betweenness'] = round(betw, 5)
            G.nodes[node_id]['pagerank'] = round(pr, 5)

            node_metadata.append({
                "id": node_id,
                "label": G.nodes[node_id]['account_id'],
                "entity_type": G.nodes[node_id]['entity_name'],
                "is_illicit": label,
                "typology": G.nodes[node_id]['typology'],
                "total_in": round(total_in, 2),
                "total_out": round(total_out, 2),
                "in_degree": k_in,
                "out_degree": k_out,
                "risk_score": G.nodes[node_id]['risk_score']
            })

        # Adjacency matrix and edge list
        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                "from": u,
                "to": v,
                "amount": data['amount'],
                "timestamp": data['timestamp'],
                "is_illicit": data['is_illicit'],
                "typology": data['typology']
            })

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        return {
            "graph": G,
            "X": X,
            "y": y,
            "nodes": node_metadata,
            "edges": edges,
            "num_illicit": sum(labels),
            "num_licit": len(labels) - sum(labels),
            "illicit_ratio": round(sum(labels) / len(labels), 4)
        }
