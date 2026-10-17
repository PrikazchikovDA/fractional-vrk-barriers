"""Polish branch-A tableaux in 50 digits (all trees below q = min(1+5a, 4+a)) and measure the local exponent
on random systems (exact step arithmetic). Log paper2/data/branchA_verify.jsonl."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src'), os.path.join(ROOT, 'paper2', 'scripts')]
import numpy as np, mpmath as mp
import polish
from verify_systems import Problem, delta_exact
mp.mp.dps = 50
T = json.load(open(os.path.join(ROOT, 'paper2', 'data', 'branchA.json')))['tableaux']
log = open(os.path.join(ROOT, 'paper2', 'data', 'branchA_verify.jsonl'), 'a')
for a_s in sys.argv[1].split(','):
    t = T[a_s]; a = float(a_s); c, A, b = np.array(t['c']), np.array(t['A']), np.array(t['b'])
    v = list(c[1:]) + [A[i, j] for i in range(1, 5) for j in range(1, i)] + list(b)
    q = min(1 + 5*a, 4 + a); am = mp.mpf(a_s)
    vm, res, nt = polish.polish(np.array(v), 5, am, q - 1e-6, iters=60)
    cm, Am, bm = polish.build(list(vm), 5, am)
    rates = []
    for seed in [0, 1, 2]:
        P = Problem(3, 4, 7, seed); errs = []
        for j in range(3, 14):
            h = mp.mpf(2)**(-j); errs.append(max(abs(x) for x in delta_exact(P, cm, Am, bm, am, h)))
        rates.append([round(float(mp.log(errs[i]/errs[i+1], 2)), 3) for i in range(len(errs)-1)])
    rec = dict(alpha=a, q_expected=q, polish_res=res, nconds=nt, rates=rates, v=[mp.nstr(x, 40) for x in vm])
    log.write(json.dumps(rec) + '\n'); log.flush()
    print(json.dumps({k: rec[k] for k in ('alpha', 'q_expected', 'polish_res', 'nconds')}), [r[-3:] for r in rates], flush=True)
