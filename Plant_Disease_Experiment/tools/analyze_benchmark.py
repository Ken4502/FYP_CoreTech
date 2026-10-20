import csv
from collections import Counter, defaultdict

from pathlib import Path

csv_path = Path(__file__).resolve().parent.parent / 'reports' / 'phytonet_500_internet_benchmark.csv'

with open(csv_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Total Evaluated Specimens: {len(rows)}")

cls_stats = defaultdict(lambda: {'total': 0, 'top1': 0, 'top3': 0, 'top5': 0})
err_cats = Counter()
water_stats = Counter()
aging_stats = Counter()
host_top1, host_top3, host_total = 0, 0, 0
univ_top1, univ_top3, univ_total = 0, 0, 0
conf_margins = []
latencies = []

confusions = Counter()

for r in rows:
    gt = r['Ground_Truth_Class']
    pred = r['PhytoNet_Top1_Prediction']
    is_host = not gt.startswith('Universal_')
    t1 = (r.get('Top1_Exact_Match', r.get('Top1_Match')) == 'YES')
    clin = (r.get('Clinical_Diagnostic_Hit', r.get('Top1_Match')) == 'YES')
    t3 = (r['Top3_Hit'] == 'YES')
    t5 = (r['Top5_Hit'] == 'YES')
    err_cats[r['Error_Category']] += 1
    water_stats[r['Abiotic_Water_Status']] += 1
    aging_stats[r['Abiotic_Aging_Status']] += 1
    conf_margins.append(float(r['Confidence_Margin_Pct']))
    latencies.append(float(r['Latency_MS']))

    cls_stats[gt]['total'] += 1
    if t1:
        cls_stats[gt]['top1'] += 1
        if is_host: host_top1 += 1
        else: univ_top1 += 1
    else:
        confusions[(gt, pred)] += 1

    if t3:
        cls_stats[gt]['top3'] += 1
        if is_host: host_top3 += 1
        else: univ_top3 += 1

    if t5:
        cls_stats[gt]['top5'] += 1

    if is_host: host_total += 1
    else: univ_total += 1

print("\n--- MACRO AGGREGATES ---")
print(f"Host-Specific ({host_total} specimens): Top-1: {host_top1/host_total*100:.2f}% ({host_top1}/{host_total}) | Top-3: {host_top3/host_total*100:.2f}% ({host_top3}/{host_total})")
print(f"Universal Taxa ({univ_total} specimens):  Top-1: {univ_top1/univ_total*100:.2f}% ({univ_top1}/{univ_total}) | Top-3: {univ_top3/univ_total*100:.2f}% ({univ_top3}/{univ_total})")
print(f"Overall Top-1: {(host_top1+univ_top1)/len(rows)*100:.2f}% | Top-3: {(host_top3+univ_top3)/len(rows)*100:.2f}% | Top-5: {sum(v['top5'] for v in cls_stats.values())/len(rows)*100:.2f}%")
print(f"Mean Latency: {sum(latencies)/len(latencies):.1f} ms | Mean Conf Margin: {sum(conf_margins)/len(conf_margins):.2f}%")

print("\n--- ERROR CATEGORIZATION ---")
for cat, cnt in err_cats.most_common():
    print(f"  {cat:<22} : {cnt:>3} ({cnt/len(rows)*100:.1f}%)")

print("\n--- ABIOTIC VIGOR DISTRIBUTION ---")
print("Water Status:", dict(water_stats))
print("Aging Status:", dict(aging_stats))

print("\n--- PER-CLASS PERFORMANCE (All 31 Classes) ---")
sorted_cls = sorted(cls_stats.items(), key=lambda x: (x[1]['top1']/x[1]['total'], x[1]['top3']/x[1]['total']), reverse=True)
for k, v in sorted_cls:
    pct1 = v['top1'] / v['total'] * 100
    pct3 = v['top3'] / v['total'] * 100
    pct5 = v['top5'] / v['total'] * 100
    print(f"  {k:<35} : Top-1: {v['top1']:>2}/{v['total']:>2} ({pct1:>5.1f}%) | Top-3: {pct3:>5.1f}% | Top-5: {pct5:>5.1f}%")

print("\n--- TOP 10 CLINICAL MISCLASSIFICATIONS ---")
for (gt, pred), cnt in confusions.most_common(10):
    print(f"  {gt}  -->  {pred} : {cnt} times")
