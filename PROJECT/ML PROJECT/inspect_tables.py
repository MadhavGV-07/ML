import json

with open('scratch_inspect.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('doc_details.txt', 'w', encoding='utf-8') as out:
    out.write('=== PLACEHOLDERS IN PARAGRAPHS ===\n')
    for item in data['placeholders']:
        out.write(f"P[{item['p_idx']}]: {item['text']}\n\n")

    out.write('=== TABLES DETAILS ===\n')
    for t in data['tables']:
        idx = t['table_idx']
        rows = t['rows']
        out.write(f"\n================ TABLE {idx} ({len(rows)} rows) ================\n")
        for r_idx, r in enumerate(rows):
            out.write(f"  Row {r_idx}: {r}\n")
print("Wrote doc_details.txt successfully")
