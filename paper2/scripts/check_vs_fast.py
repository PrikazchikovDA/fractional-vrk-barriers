"""Check the tree B-series against the independent series generator src/vrk_fast.py (scalar).
For random tableaux: every coefficient of the series (exponent pair (N,K), scalar atom monomial) must
equal the sum of Psi(T)/sigma(T) over trees with that monomial. Also compares the counts of conditions."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'src'), os.path.join(ROOT, 'paper2', 'src')]
import numpy as np
from collections import defaultdict
from vrk_fast import Setup
from vtrees import all_trees, NK, atoms, sigma, Psi, Tab, unpack

def compare(s, alpha, p, seed):
    S = Setup(s, alpha, p); rng = np.random.default_rng(seed)
    v = rng.uniform(0.1, 1.2, S.nunk())
    d = S.delta(v)
    c, A, b = unpack(v, s, alpha); T = Tab(c, A, b, alpha)
    tr = defaultdict(float)
    for t in all_trees(alpha, p):
        N, K = NK(t); tr[(N, K, atoms(t))] += Psi(T, t) / sigma(t)
    keys = set(d) | set(tr)
    err = max(abs(d.get(kk, 0.0) - tr.get(kk, 0.0)) for kk in keys)
    scale = max(abs(x) for x in tr.values())
    return err, scale, len(keys)

if __name__ == '__main__':
    out = []
    for s, alpha, p in [(3, 0.3, 3.3), (4, 0.7, 4.5), (5, 0.6, 4.4), (5, 0.37, 4.2), (6, 0.45, 4.0), (4, 0.5, 4.0)]:
        for seed in range(3):
            err, sc, n = compare(s, alpha, p, seed)
            rec = dict(s=s, alpha=alpha, p=p, seed=seed, max_abs_diff=err, max_coef=sc, nterms=n)
            print(json.dumps(rec), flush=True); out.append(rec)
    json.dump(out, open(os.path.join(ROOT, 'paper2', 'data', 'check_vs_fast.json'), 'w'), indent=1)
