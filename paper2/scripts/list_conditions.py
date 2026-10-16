"""List all trees (order conditions) with given (N,K) levels, in matrix notation; writes markdown + json."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src')]
from collections import defaultdict
from vtrees import trees_by_order, NK, sigma, weight_expr, diff_expr, atoms, height
levels = defaultdict(list)
for n in range(1, 10):
    for t in trees_by_order(n):
        N, K = NK(t)
        if N + K <= 7: levels[(N, K)].append(t)
md = []; js = []
def expo(N, K): return (f'{K}+' if K else '') + (f'{N}α' if N > 1 else 'α')
for (N, K) in sorted(levels, key=lambda x: (x[0]+x[1], x[1])):
    ts = sorted(levels[(N, K)], key=weight_expr)
    groups = defaultdict(list)
    for t in ts: groups[atoms(t)].append(t)
    md.append(f'\n### h^({expo(N,K)}),  (N,K)=({N},{K}): {len(ts)} условий для систем, {len(groups)} для скаляра\n')
    md.append('| # | элементарный дифференциал | вес Ψ(T) | σ | стад. высота | скаляр. группа |\n|---|---|---|---|---|---|')
    gid = {a: i+1 for i, a in enumerate(groups)}
    for i, t in enumerate(ts):
        md.append(f'| {i+1} | {diff_expr(t)} | {weight_expr(t)} | {sigma(t)} | {height(t)} | {gid[atoms(t)]} |')
        js.append(dict(N=N, K=K, diff=diff_expr(t), weight=weight_expr(t), sigma=sigma(t), height=height(t), scalar_group=gid[atoms(t)]))
open(os.path.join(ROOT, 'paper2', 'data', 'conditions_N+K_le_7.md'), 'w').write('\n'.join(md))
json.dump(js, open(os.path.join(ROOT, 'paper2', 'data', 'conditions_N+K_le_7.json'), 'w'), ensure_ascii=False, indent=0)
print('\n'.join(md))
