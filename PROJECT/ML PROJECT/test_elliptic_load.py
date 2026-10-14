import csv, os, time

p = r'C:\Users\Madhav\Downloads\elliptic_bitcoin_dataset'
t0 = time.time()

# Load classes
classes = {}
with open(os.path.join(p, 'elliptic_txs_classes.csv'), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader) # skip header: txId, class
    for row in reader:
        # class: '1'=illicit, '2'=licit, 'unknown'=unknown
        classes[row[0]] = row[1]

print(f"Loaded {len(classes)} classes in {time.time()-t0:.2f}s")
counts = {}
for c in classes.values():
    counts[c] = counts.get(c, 0) + 1
print("Counts:", counts)
