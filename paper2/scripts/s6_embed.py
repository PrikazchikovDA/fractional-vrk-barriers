"""s=6 barrier evidence via warm starts from embedded s=5 S-family methods (stage 6 with b6=0) plus random LM.
For each alpha: raise level L through s=6 tree exponents; at each level try warm start from previous level,
then 20 perturbed embeddings, then 40 random starts. Restartable log paper2/data/s6_embed.jsonl."""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src')]
import numpy as np
from scipy.optimize import least_squares
from vtrees import unpack
from search import Conds, search_lm, _to_v, _to_z
from barrier_map_levels import levels
def pack(c, A, b, s):
    v = list(c[1:s])
    for i in range(1, s):
        for j in range(1, i): v.append(A[i, j])
    return np.array(v + list(b))
def lm(C, s, z0, nfev=3000):
    n = C.nunk; pad = max(0, n - C.ncond)
    F = lambda z: np.concatenate([C.residual(_to_v(z, s, 1.5)), np.zeros(pad)])
    return least_squares(F, z0, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=nfev)
pts = {round(json.loads(l)['alpha'], 4): json.loads(l) for l in open(os.path.join(ROOT, 'paper2', 'data', 'cont_S16.jsonl')) if json.loads(l)['res'] < 1e-11}
path = os.path.join(ROOT, 'paper2', 'data', 's6_embed.jsonl'); done = set()
if os.path.exists(path):
    for l in open(path): r = json.loads(l); done.add((r['alpha'], r['L']))
log = open(path, 'a'); rng = np.random.default_rng(1)
for a in [0.6, 0.7, 0.8, 0.9, 0.5, 0.75, 0.85, 0.65, 0.55, 0.95]:
    c5, A5, b5 = unpack(np.array(pts[a]['x']), 5, a)
    embeds = []
    for k in range(20):
        c = np.append(c5, rng.uniform(0.2, 1.4)); A = np.zeros((6, 6)); A[:5, :5] = A5
        A[5, 1:5] = rng.normal(scale=0.5, size=4); b = np.append(b5, 0.0)
        embeds.append(_to_z(pack(c, A, b, 6), 6, 1.5))
    warm = None
    for L in levels(6, a, 7.5):
        if (a, L) in done: continue
        t0 = time.time(); C = Conds(6, a, L + 1e-7, 'system'); best = None; how = None
        starts = ([('warm', warm)] if warm is not None else []) + [('embed', z) for z in embeds]
        for kind, z0 in starts:
            r = lm(C, 6, z0)
            if best is None or r.cost < best.cost: best, how = r, kind
            if np.sqrt(2*best.cost) < 1e-11: break
        res = float(np.sqrt(2*best.cost))
        if res >= 1e-11:
            rr = search_lm(6, a, L, 'system', 1.5, tries=40, seed=int(1000*a + 10*L))
            if rr['res'] < res: res, how, x = rr['res'], 'random', np.array(rr['x'])
            else: x = _to_v(best.x, 6, 1.5)
        else: x = _to_v(best.x, 6, 1.5)
        rec = dict(alpha=a, L=L, found=res < 1e-11, res=res, how=how, ncond=C.ncond, maxcoef=float(np.abs(x).max()), sec=round(time.time()-t0, 1), x=x.tolist())
        log.write(json.dumps(rec) + '\n'); log.flush(); print(json.dumps({k: v for k, v in rec.items() if k != 'x'}), flush=True)
        if not rec['found']: break
        warm = _to_z(x, 6, 1.5)
