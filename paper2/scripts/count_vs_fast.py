"""Compare numbers of conditions: tree enumeration vs src/vrk_fast.py keys.
vrk_fast lists every atom monomial reachable by the recursion, including monomials whose coefficient is
identically zero because d_0 = 0 (row-sum condition) or c_1 = 0. 'fast_nonzero' counts keys whose
coefficient is nonzero at a random tableau (generic rank of the key set)."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'src'), os.path.join(ROOT, 'paper2', 'src')]
from fractions import Fraction
import numpy as np
from vrk_fast import Setup
from vtrees import count_conditions
rows = []
for s, a, p in [(2,'3/10','8/5'),(3,'3/10','23/10'),(4,'3/10','13/5'),(4,'7/10','19/5'),(5,'7/10','22/5'),(5,'7/10','21/5'),
                (5,'3/10','29/10'),(5,'3/5','4'),(5,'4/5','23/5'),(5,'9/10','24/5'),(6,'3/5','23/5'),(6,'3/10','16/5'),(5,'1/2','7/2'),(6,'1/2','4')]:
    af, pf = float(Fraction(a)), float(Fraction(p))
    S = Setup(s, af, pf); ks = S.keys(); nz = 0
    for seed in range(3):
        v = np.random.default_rng(seed).uniform(0.1, 1.2, S.nunk())
        nz = max(nz, int((np.abs(S.residual(v, ks)) > 1e-13).sum()))
    sc = count_conditions(s, af, pf, True); sy = count_conditions(s, af, pf, False)
    r = dict(s=s, alpha=a, p=p, fast_keys=len(ks), fast_nonzero=nz, trees_scalar=sc, trees_system=sy, unknowns=S.nunk())
    rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open(os.path.join(ROOT, 'paper2', 'data', 'count_vs_fast.json'), 'w'), indent=1)
