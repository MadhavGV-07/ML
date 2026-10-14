"""
Flask Web Application & REST API Server for Anti-Money Laundering GNN Platform.
Serves the modern cyber-fintech dashboard and powers live GNN inference,
training, XAI graph explanations, dynamic transaction ledgers, and forensic audits.
"""

import os
import io
import csv
import json
import random
import numpy as np
import networkx as nx
from flask import Flask, render_template, request, jsonify, send_file
from engine import AMLEngine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aml-gnn-fintech-security-token-2026'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB max upload

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global instance of the AML GNN Engine
engine = AMLEngine(seed=42)

# ---------------------------------------------------------------------------
# Column role auto-detection
# ---------------------------------------------------------------------------
COLUMN_ROLES = {
    'sender': [
        'sender_account', 'sender', 'source', 'from', 'src', 'payer',
        'originator', 'account_from', 'account_sender', 'from_account',
        'nameorig', 'origin', 'cust_id', 'customer_id', 'client_id',
        'source_id', 'src_id', 'card_id', 'user_id', 'orig_account',
        'account', 'senderaccount', 'sourceaccount'
    ],
    'receiver': [
        'receiver_account', 'receiver', 'target', 'to', 'dst', 'beneficiary',
        'payee', 'destination', 'account_to', 'account_receiver', 'to_account',
        'namedest', 'dest', 'dest_id', 'merchant_id', 'merchant', 'target_id',
        'dst_id', 'counterparty', 'receiveraccount', 'targetaccount'
    ],
    'amount': [
        'amount', 'value', 'sum', 'transaction_amount', 'amt', 'txn_amount',
        'transfer_amount', 'usd_amount', 'base_amt', 'volume', 'tx_amount'
    ],
    'label': [
        'is_laundering', 'is_illicit', 'label', 'fraud', 'suspicious',
        'is_fraud', 'flag', 'class', 'target_label', 'isfraud', 'isflaggedfraud',
        'illicit', 'aml_alert', 'is_suspicious'
    ],
    'typology': [
        'laundering_type', 'typology', 'type', 'category', 'pattern',
        'fraud_type', 'crime_type', 'trans_type', 'transaction_type', 'channel',
        'tag', 'scenario'
    ],
    'currency': [
        'payment_currency', 'currency', 'ccy', 'received_currency',
        'from_currency', 'to_currency', 'orig_currency', 'dest_currency'
    ],
    'payment_type': [
        'payment_type', 'payment_method', 'method', 'txn_type',
        'transaction_type', 'channel', 'mode', 'transfer_type'
    ],
    'sender_location': [
        'sender_bank_location', 'sender_location', 'from_country',
        'from_location', 'origin_country', 'sender_country', 'orig_country'
    ],
    'receiver_location': [
        'receiver_bank_location', 'receiver_location', 'to_country',
        'to_location', 'dest_country', 'receiver_country'
    ],
    'date': [
        'date', 'transaction_date', 'txn_date', 'datetime', 'created_at',
        'trans_date', 'timestamp_date'
    ],
    'time': [
        'time', 'transaction_time', 'txn_time', 'timestamp', 'step', 'hour'
    ],
}

def detect_column_roles(headers):
    """Map CSV column headers to semantic roles using fuzzy matching."""
    normalized = {h: h.lower().strip().replace(' ', '_').replace('-', '_') for h in headers}
    mapping = {}

    # Exact match in candidate list
    for role, candidates in COLUMN_ROLES.items():
        for col, norm in normalized.items():
            if norm in candidates:
                if role not in mapping:
                    mapping[role] = col
                break

    # Substring / partial match fallback for unmapped roles
    for role, candidates in COLUMN_ROLES.items():
        if role in mapping:
            continue
        for col, norm in normalized.items():
            if any(cand in norm for cand in candidates if len(cand) >= 4):
                mapping[role] = col
                break

    return mapping


def detect_delimiter(sample_text):
    """Detects delimiter from a text sample."""
    for delim in [',', ';', '\t', '|']:
        if delim in sample_text and sample_text.count(delim) > 2:
            return delim
    try:
        dialect = csv.Sniffer().sniff(sample_text[:2048])
        return dialect.delimiter
    except Exception:
        return ','


def build_dataset_from_stream(stream_or_path, col_map, max_nodes=2500, scan_limit=250000):
    """
    High-speed, memory-efficient streaming CSV parser.
    Reads row-by-row up to scan_limit, collects stratified illicit + licit transactions,
    preserves all CSV metadata and custom features, and computes graph metrics.
    """
    sender_col   = col_map.get('sender')
    receiver_col = col_map.get('receiver')
    amount_col   = col_map.get('amount')
    label_col    = col_map.get('label')
    typology_col = col_map.get('typology')
    currency_col = col_map.get('currency')
    pay_type_col = col_map.get('payment_type')
    s_loc_col    = col_map.get('sender_location')
    r_loc_col    = col_map.get('receiver_location')
    date_col     = col_map.get('date')
    time_col     = col_map.get('time')

    if not sender_col or not receiver_col:
        raise ValueError("Could not detect sender and receiver columns. Please specify them in the column mapper.")

    # Open file or use stream
    is_path = isinstance(stream_or_path, str)
    f = open(stream_or_path, 'r', encoding='utf-8-sig', errors='replace') if is_path else stream_or_path

    try:
        # Sniff delimiter
        pos = f.tell() if hasattr(f, 'tell') else 0
        sample_chunk = f.read(4096)
        delim = detect_delimiter(sample_chunk)
        if hasattr(f, 'seek'):
            f.seek(pos)

        reader = csv.DictReader(f, delimiter=delim)
        headers = reader.fieldnames or []

        illicit_rows = []
        licit_rows = []
        total_scanned = 0
        max_illicit = max(20, int(max_nodes * 0.25))
        max_licit = max_nodes - max_illicit

        random.seed(42)

        for row in reader:
            total_scanned += 1
            u_raw = str(row.get(sender_col, '') or '').strip()
            v_raw = str(row.get(receiver_col, '') or '').strip()
            if not u_raw or not v_raw or u_raw == v_raw:
                continue

            # Determine illicit status
            is_ill = 0
            if label_col:
                raw_lbl = str(row.get(label_col, '0')).strip().lower()
                if raw_lbl in ('1', '1.0', 'true', 'yes', 'fraud', 'illicit', 'suspicious'):
                    is_ill = 1

            if is_ill == 1:
                if len(illicit_rows) < max_illicit:
                    illicit_rows.append(row)
                else:
                    # Reservoir sampling for illicit
                    idx = random.randint(0, len(illicit_rows))
                    if idx < max_illicit:
                        illicit_rows[idx] = row
            else:
                if len(licit_rows) < max_licit:
                    licit_rows.append(row)
                else:
                    # Reservoir sampling for licit
                    idx = random.randint(0, total_scanned)
                    if idx < max_licit:
                        licit_rows[idx] = row

            if total_scanned >= scan_limit:
                break

    finally:
        if is_path:
            f.close()

    # Combine sampled rows
    sampled_rows = illicit_rows + licit_rows
    random.shuffle(sampled_rows)

    if not sampled_rows:
        raise ValueError("No valid transactions found in CSV. Ensure sender and receiver contain distinct account IDs.")

    # ── Build raw directed graph ──────────────────────────────────────────
    nodes_set = set()
    all_currencies = set()
    all_pay_types  = set()
    all_locations  = set()

    for r in sampled_rows:
        u = str(r.get(sender_col, '') or '').strip()
        v = str(r.get(receiver_col, '') or '').strip()
        if not u or not v or u == v:
            continue
        nodes_set.add(u)
        nodes_set.add(v)
        if currency_col:
            all_currencies.add(str(r.get(currency_col, 'USD') or 'USD').strip())
        if pay_type_col:
            all_pay_types.add(str(r.get(pay_type_col, 'Unknown') or 'Unknown').strip())
        if s_loc_col:
            all_locations.add(str(r.get(s_loc_col, 'Unknown') or 'Unknown').strip())
        if r_loc_col:
            all_locations.add(str(r.get(r_loc_col, 'Unknown') or 'Unknown').strip())

    # Map node labels to integers
    node_map = {name: i for i, name in enumerate(sorted(nodes_set))}
    inv_map = {i: name for name, i in node_map.items()}

    G = nx.DiGraph()
    for name, idx in node_map.items():
        G.add_node(idx, account_id=str(name), label=str(name))

    for r in sampled_rows:
        u_raw = str(r.get(sender_col, '') or '').strip()
        v_raw = str(r.get(receiver_col, '') or '').strip()
        if not u_raw or not v_raw or u_raw == v_raw:
            continue
        if u_raw not in node_map or v_raw not in node_map:
            continue
        u, v = node_map[u_raw], node_map[v_raw]

        try:
            amt_str = str(r.get(amount_col, 100.0) or 100.0).replace('$', '').replace(',', '').strip()
            amount = float(amt_str)
        except (ValueError, TypeError):
            amount = 100.0

        is_ill = 0
        if label_col:
            raw_lbl = str(r.get(label_col, '0')).strip().lower()
            if raw_lbl in ('1', '1.0', 'true', 'yes', 'fraud', 'illicit', 'suspicious'):
                is_ill = 1

        typology = 'Normal'
        if typology_col:
            typology = str(r.get(typology_col, 'Normal') or 'Normal').strip()
        elif is_ill == 1:
            typology = 'Flagged Illicit'

        currency  = str(r.get(currency_col, 'USD') or 'USD').strip() if currency_col else 'USD'
        pay_type  = str(r.get(pay_type_col, 'Transfer') or 'Transfer').strip() if pay_type_col else 'Transfer'
        s_loc     = str(r.get(s_loc_col, 'Domestic') or 'Domestic').strip() if s_loc_col else 'Domestic'
        r_loc     = str(r.get(r_loc_col, 'Domestic') or 'Domestic').strip() if r_loc_col else 'Domestic'
        d_val     = str(r.get(date_col, '2026-03-01') or '—').strip() if date_col else '—'
        t_val     = str(r.get(time_col, '12:00:00') or '—').strip() if time_col else '—'

        # Store all other columns as custom features
        custom_feats = {k: v for k, v in r.items() if k not in (sender_col, receiver_col) and v is not None}

        G.add_edge(u, v,
                   amount=amount,
                   timestamp=1,
                   date=d_val,
                   time=t_val,
                   currency=currency,
                   pay_type=pay_type,
                   payment_type=pay_type,
                   sender_loc=s_loc,
                   sender_location=s_loc,
                   receiver_loc=r_loc,
                   receiver_location=r_loc,
                   is_illicit=is_ill,
                   typology=typology,
                   features=custom_feats)

    N = G.number_of_nodes()
    if N == 0:
        raise ValueError("No valid nodes could be constructed from the dataset.")

    # ── Compute graph metrics ─────────────────────────────────────────────
    try:
        bet = nx.betweenness_centrality(G, k=min(80, N), normalized=True, seed=42) if N > 300 else nx.betweenness_centrality(G, normalized=True)
    except Exception:
        bet = {n: 0.0 for n in G.nodes()}

    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100)
    except Exception:
        pr = {n: 1.0 / N for n in G.nodes()}

    # ── Build node aggregates ─────────────────────────────────────────────
    node_total_in   = {n: 0.0 for n in G.nodes()}
    node_total_out  = {n: 0.0 for n in G.nodes()}
    node_currencies = {n: set() for n in G.nodes()}
    node_locations  = {n: set() for n in G.nodes()}
    node_pay_types  = {n: set() for n in G.nodes()}
    node_is_illicit = {n: 0 for n in G.nodes()}
    node_typology   = {n: 'Licit Retail' for n in G.nodes()}
    node_amounts_in = {n: [] for n in G.nodes()}

    for u, v, d in G.edges(data=True):
        amt = d.get('amount', 0.0)
        node_total_out[u] += amt
        node_total_in[v]  += amt
        node_currencies[u].add(d.get('currency', 'USD'))
        node_currencies[v].add(d.get('currency', 'USD'))
        node_locations[u].add(d.get('sender_loc', 'Domestic'))
        node_locations[v].add(d.get('receiver_loc', 'Domestic'))
        node_pay_types[u].add(d.get('pay_type', 'Transfer'))
        node_pay_types[v].add(d.get('pay_type', 'Transfer'))
        node_amounts_in[v].append(amt)
        if d.get('is_illicit', 0) == 1:
            node_is_illicit[u] = 1
            node_is_illicit[v] = 1
            node_typology[u] = d.get('typology', 'Flagged')
            node_typology[v] = d.get('typology', 'Flagged')

    # If dataset was completely unlabeled, generate heuristic anomaly pseudo-labels so GNN can train
    if sum(node_is_illicit.values()) == 0:
        # Flag top 5% by betweenness / PageRank as anomaly candidates
        threshold_pr = np.percentile(list(pr.values()), 95)
        for n in G.nodes():
            if pr[n] >= threshold_pr:
                node_is_illicit[n] = 1
                node_typology[n] = 'High Flow Anomaly'

    sorted_nodes = sorted(G.nodes())

    # ── Build 15-dim feature matrix X ─────────────────────────────────────
    X_rows = []
    for n in sorted_nodes:
        in_deg  = G.in_degree(n)
        out_deg = G.out_degree(n)
        t_in    = node_total_in[n]
        t_out   = node_total_out[n]
        amounts = node_amounts_in[n]
        avg_in  = float(np.mean(amounts)) if amounts else 0.0
        max_in  = float(np.max(amounts))  if amounts else 0.0
        log_vol = float(np.log1p(t_in + t_out))
        ratio   = t_in / (t_out + 1e-9)
        n_ill   = sum(1 for nb in G.predecessors(n) if node_is_illicit.get(nb, 0) == 1)
        n_ill  += sum(1 for nb in G.successors(n)   if node_is_illicit.get(nb, 0) == 1)
        cur_div = float(len(node_currencies[n]))
        loc_div = float(len(node_locations[n]))
        pay_div = float(len(node_pay_types[n]))

        feats = [
            t_in / 1e6,
            t_out / 1e6,
            float(in_deg) / max(1, N),
            float(out_deg) / max(1, N),
            float(bet.get(n, 0.0)),
            float(pr.get(n, 0.0)),
            cur_div / max(1, len(all_currencies) or 1),
            loc_div / max(1, len(all_locations) or 1),
            pay_div / max(1, len(all_pay_types) or 1),
            avg_in / 1e5,
            max_in / 1e5,
            log_vol / 20.0,
            min(ratio, 10.0) / 10.0,
            float(n_ill) / max(1, in_deg + out_deg),
            float(random.uniform(0.1, 1.0))
        ]
        X_rows.append(feats)

    X = np.array(X_rows, dtype=np.float32)
    X = np.clip(X, -10, 10)
    y = np.array([node_is_illicit[n] for n in sorted_nodes], dtype=np.int64)

    # ── Build nodes & edges list for frontend & dossiers ─────────────────
    nodes_list = []
    for n in sorted_nodes:
        in_deg  = G.in_degree(n)
        out_deg = G.out_degree(n)
        vol = node_total_in[n] + node_total_out[n]
        if vol > 250000 or (in_deg + out_deg) >= 8:
            entity_type = 'Commercial Entity'
        elif len(node_currencies[n]) >= 2:
            entity_type = 'Crypto Exchange'
        else:
            entity_type = 'Retail Account'

        risk = float(bet.get(n, 0.0)) * 5.0 + float(pr.get(n, 0.0)) * 3.0
        risk = min(risk + (0.35 if node_is_illicit[n] else 0.03), 0.98)

        # Store attributes on networkx node as well
        G.nodes[n]['account_id'] = str(inv_map[n])
        G.nodes[n]['entity_name'] = entity_type
        G.nodes[n]['entity_type'] = entity_type
        G.nodes[n]['is_illicit'] = node_is_illicit[n]
        G.nodes[n]['typology'] = node_typology[n]
        G.nodes[n]['risk_score'] = round(risk, 4)
        G.nodes[n]['total_in'] = round(node_total_in[n], 2)
        G.nodes[n]['total_out'] = round(node_total_out[n], 2)
        G.nodes[n]['in_degree'] = in_deg
        G.nodes[n]['out_degree'] = out_deg
        G.nodes[n]['betweenness'] = round(float(bet.get(n, 0.0)), 6)
        G.nodes[n]['pagerank'] = round(float(pr.get(n, 0.0)), 6)

        nodes_list.append({
            'id': n,
            'label': str(inv_map[n]),
            'account_id': str(inv_map[n]),
            'entity_type': entity_type,
            'entity_name': entity_type,
            'is_illicit': node_is_illicit[n],
            'typology': node_typology[n],
            'risk_score': round(risk, 4),
            'total_in': round(node_total_in[n], 2),
            'total_out': round(node_total_out[n], 2),
            'in_degree': in_deg,
            'out_degree': out_deg,
            'betweenness': round(float(bet.get(n, 0.0)), 6),
            'pagerank': round(float(pr.get(n, 0.0)), 6),
            'tenure_days': 365,
            'pred_risk_score': round(risk, 4),
            'pred_is_illicit': node_is_illicit[n],
        })

    edges_list = []
    for u, v, d in G.edges(data=True):
        edges_list.append({
            'from': u,
            'to': v,
            'from_account': G.nodes[u]['account_id'],
            'to_account': G.nodes[v]['account_id'],
            'amount': round(d.get('amount', 0.0), 2),
            'timestamp': d.get('timestamp', 1),
            'date': d.get('date', '—'),
            'time': d.get('time', '—'),
            'payment_type': d.get('pay_type', 'Transfer'),
            'currency': d.get('currency', 'USD'),
            'sender_loc': d.get('sender_loc', 'Domestic'),
            'receiver_loc': d.get('receiver_loc', 'Domestic'),
            'is_illicit': d.get('is_illicit', 0),
            'typology': d.get('typology', 'Normal'),
            'custom_features': d.get('features', {})
        })

    num_illicit = int(y.sum())
    dataset = {
        'graph':       G,
        'X':           X,
        'y':           y,
        'nodes':       nodes_list,
        'edges':       edges_list,
        'num_illicit': num_illicit,
        'num_licit':   N - num_illicit,
        'illicit_ratio': round(num_illicit / max(1, N), 4),
    }

    sample_info = {
        'total_scanned_rows': total_scanned,
        'transactions_sampled': len(sampled_rows),
        'unique_accounts': N,
        'unique_transactions': len(edges_list),
        'detected_illicit': num_illicit,
        'has_labels': label_col is not None,
        'column_mapping': col_map,
    }

    return dataset, sample_info


# ===========================================================================
# Routes
# ===========================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/graph', methods=['GET'])
def get_graph():
    summary = engine.get_graph_summary()
    return jsonify(summary)

@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.get_json() or {}
    model_name = data.get('model_name', 'GAT')
    epochs = int(data.get('epochs', 100))
    lr = float(data.get('lr', 0.015))
    loss_type = data.get('loss_type', 'focal')

    try:
        res = engine.train_model(model_name=model_name, epochs=epochs, lr=lr, loss_type=loss_type)
        return jsonify({"status": "success", "result": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    threshold = float(data.get('threshold', 0.50))
    model_name = data.get('model_name', engine.active_model_name)

    try:
        metrics = engine.evaluate_model(model_name=model_name, threshold=threshold)
        return jsonify({"status": "success", "metrics": metrics, "nodes": engine.current_dataset["nodes"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/explain/<int:node_id>', methods=['GET'])
def explain_node(node_id):
    try:
        explanation = engine.explain_node(node_id)
        return jsonify({"status": "success", "explanation": explanation})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/transaction/<int:u>/<int:v>', methods=['GET'])
def get_transaction(u, v):
    try:
        details = engine.get_edge_details(u, v)
        if "error" in details:
            return jsonify({"status": "error", "message": details["error"]}), 404
        return jsonify({"status": "success", "transaction": details})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """Returns all transactions for Global Transaction Explorer."""
    try:
        limit = int(request.args.get('limit', 150))
        offset = int(request.args.get('offset', 0))
        illicit_only = request.args.get('illicit_only', 'false').lower() == 'true'
        search = request.args.get('search', '').strip().lower()

        G = engine.current_dataset["graph"]
        txns = []
        for u, v, d in G.edges(data=True):
            u_acc = G.nodes[u].get('account_id', str(u))
            v_acc = G.nodes[v].get('account_id', str(v))
            is_ill = d.get('is_illicit', 0)
            typology = d.get('typology', 'Normal')

            if illicit_only and is_ill != 1:
                continue
            if search:
                match = (search in u_acc.lower() or search in v_acc.lower() or search in typology.lower() or search in str(d.get('pay_type', '')).lower())
                if not match:
                    continue

            txns.append({
                "from_node": u,
                "to_node": v,
                "from_account": u_acc,
                "to_account": v_acc,
                "amount": round(float(d.get('amount', 0.0)), 2),
                "currency": d.get('currency', 'USD'),
                "payment_type": d.get('pay_type', d.get('payment_type', 'Transfer')),
                "date": d.get('date', '—'),
                "time": d.get('time', '—'),
                "sender_loc": d.get('sender_loc', 'Domestic'),
                "receiver_loc": d.get('receiver_loc', 'Domestic'),
                "typology": typology,
                "is_illicit": is_ill,
                "features": d.get('features', {})
            })

        total = len(txns)
        txns.sort(key=lambda x: x['amount'], reverse=True)
        return jsonify({
            "status": "success",
            "total": total,
            "transactions": txns[offset:offset+limit]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/benchmark', methods=['GET'])
def benchmark():
    try:
        benchmarks = engine.benchmark_comparison()
        return jsonify({"status": "success", "benchmarks": benchmarks})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_benchmark():
    data = request.get_json() or {}
    num_nodes = int(data.get('num_nodes', 260))
    illicit_ratio = float(data.get('illicit_ratio', 0.08))

    try:
        summary = engine.load_benchmark_dataset(num_nodes=num_nodes, illicit_ratio=illicit_ratio)
        engine.train_model(model_name="GAT", epochs=70)
        return jsonify({"status": "success", "summary": summary})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/sar/<int:node_id>', methods=['GET'])
def export_sar(node_id):
    try:
        sar = engine.generate_sar_report(node_id)
        as_attachment = request.args.get('download', 'false').lower() == 'true'
        if as_attachment:
            buf = io.BytesIO()
            buf.write(json.dumps(sar, indent=2).encode('utf-8'))
            buf.seek(0)
            return send_file(
                buf,
                mimetype='application/json',
                as_attachment=True,
                download_name=f"FinCEN_SAR_{sar.get('report_id', node_id)}.json"
            )
        return jsonify({"status": "success", "sar": sar})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Instant Load Local SAML-D.csv (Pre-installed 1GB Dataset)
# ---------------------------------------------------------------------------
@app.route('/api/load-local-saml', methods=['POST'])
def load_local_saml():
    """
    Directly streams and samples from the local SAML-D.csv file in seconds.
    Avoids 1GB browser upload transfer completely.
    """
    data = request.get_json() or {}
    max_nodes = int(data.get('max_nodes', 1200))
    saml_path = os.path.join(os.path.dirname(__file__), 'SAML-D.csv')

    if not os.path.exists(saml_path):
        return jsonify({"status": "error", "message": "SAML-D.csv not found in workspace."}), 404

    try:
        # Detect header from first row
        with open(saml_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            line = f.readline()
            delim = detect_delimiter(line)
            headers = [h.strip() for h in line.split(delim)]

        col_map = detect_column_roles(headers)
        dataset, sample_info = build_dataset_from_stream(saml_path, col_map, max_nodes=max_nodes, scan_limit=350000)

        engine.load_from_parsed_data(dataset)
        engine.train_model(model_name="GAT", epochs=60)

        summary = engine.get_graph_summary()
        summary['sample_info'] = sample_info
        return jsonify({"status": "success", "summary": summary})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to load SAML-D.csv: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# CSV Upload — Step 1: detect columns (fast, reads first few KB only)
# ---------------------------------------------------------------------------
@app.route('/api/upload-csv/columns', methods=['POST'])
def detect_csv_columns():
    """Reads only the first 64KB chunk of the uploaded file to detect headers & roles."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({"status": "error", "message": "No file selected"}), 400

    try:
        # Save file to uploads folder so step 2 can parse directly without re-uploading
        saved_path = os.path.join(UPLOAD_FOLDER, 'current_upload.csv')
        file.save(saved_path)

        with open(saved_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            chunk = f.read(65536)
            delim = detect_delimiter(chunk)
            f.seek(0)
            reader = csv.DictReader(f, delimiter=delim)
            headers = reader.fieldnames or []
            sample_rows = []
            for i, row in enumerate(reader):
                if i >= 5:
                    break
                sample_rows.append(dict(row))

        col_map = detect_column_roles(headers)
        return jsonify({
            "status": "success",
            "headers": list(headers),
            "detected_mapping": col_map,
            "sample_rows": sample_rows,
            "delimiter": delim,
            "file_cached": True
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Column detection error: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# CSV Upload — Step 2: full import with streaming parser
# ---------------------------------------------------------------------------
@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """
    Imports the dataset using the streaming parser.
    Uses either the cached upload file or the newly provided file stream.
    """
    saved_path = os.path.join(UPLOAD_FOLDER, 'current_upload.csv')

    # Check if a new file was sent or if we can use the cached upload
    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        file.save(saved_path)
    elif not os.path.exists(saved_path):
        return jsonify({"status": "error", "message": "No uploaded file available to import"}), 400

    # Parse optional user column mapping overrides
    user_col_map = {}
    col_map_json = request.form.get('col_map', '')
    if col_map_json:
        try:
            user_col_map = json.loads(col_map_json)
        except Exception:
            pass

    max_nodes = int(request.form.get('max_nodes', 2500))

    try:
        with open(saved_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            chunk = f.read(4096)
            delim = detect_delimiter(chunk)
            f.seek(0)
            reader = csv.DictReader(f, delimiter=delim)
            headers = reader.fieldnames or []

        col_map = detect_column_roles(headers)
        col_map.update({k: v for k, v in user_col_map.items() if v})

        dataset, sample_info = build_dataset_from_stream(saved_path, col_map, max_nodes=max_nodes, scan_limit=300000)

        engine.load_from_parsed_data(dataset)
        engine.train_model(model_name="GAT", epochs=60)

        summary = engine.get_graph_summary()
        summary['sample_info'] = sample_info
        return jsonify({"status": "success", "summary": summary})

    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Import error: {str(e)}"}), 500


if __name__ == '__main__':
    print("[AML Engine] Initializing default GAT model...")
    try:
        engine.train_model(model_name="GAT", epochs=60)
        print("[AML Engine] Default GAT model initialized successfully.")
    except Exception as e:
        print(f"[AML Engine] Warmup warning: {e}")

    app.run(host='127.0.0.1', port=5000, debug=False)
