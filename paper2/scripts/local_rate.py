"""Direct measurement of the local error exponent of a given tableau on random systems from the
free-atom construction (exact arithmetic of the step, mpmath; lag term cancels by construction).
usage: python local_rate.py tableau.npy s alpha [seeds]"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src'), os.path.join(ROOT, 'paper2', 'scripts')]
import numpy as np, mpmath as mp
from vtrees import unpack
from verify_systems import Problem, delta_exact
mp.mp.dps = 50
if __name__ == '__main__':
    s = int(sys.argv[2]); a = float(sys.argv[3])
    if sys.argv[1].endswith('.json'):
        vm = [mp.mpf(x) for x in json.load(open(sys.argv[1]))['v']]; v = np.array([float(x) for x in vm])
    else:
        v = np.load(sys.argv[1]); vm = [mp.mpf(float(x)) for x in v]
    seeds = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [0, 1]
    from polish import build
    am = mp.mpf(sys.argv[3]); cm, Am, bm = build(vm, s, am)
    for seed in seeds:
        P = Problem(3, 4, 7, seed); errs = []
        for j in range(3, 15):
            h = mp.mpf(2)**(-j); errs.append(max(abs(x) for x in delta_exact(P, cm, Am, bm, am, h)))
        rates = [round(float(mp.log(errs[i]/errs[i+1], 2)), 3) for i in range(len(errs)-1)]
        rec = dict(tableau=os.path.basename(sys.argv[1]), s=s, alpha=a, seed=seed, rates=rates)
        print(json.dumps(rec), flush=True)
        open(os.path.join(ROOT, 'paper2', 'data', 'local_rate.jsonl'), 'a').write(json.dumps(rec) + '\n')
