"""Restartable barrier map: for each alpha, raise the level L through tree exponents and test whether
all conditions with |T| <= L can be zeroed. q(s,alpha) = first L for which the search fails
(numerical evidence only). Log: paper2/data/map_s{s}_{mode}_{cmax}.jsonl, one record per (alpha, L).
usage: python barrier_map.py s mode cmax alpha1,alpha2,...  [tries]"""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src')]
import numpy as np
from search import Conds, search, search_lm
from scipy.optimize import least_squares
from vtrees import all_trees, NK, height

def levels(s, a, top):
    return sorted({round(NK(t)[0]*a + NK(t)[1], 10) for t in all_trees(a, top) if height(t) <= s})

def try_level(s, a, L, mode, cmax, tries, warm):
    return search_lm(s, a, L, mode, cmax, tries=tries, seed=int(1000*a) + int(100*L), warm=warm)

if __name__ == '__main__':
    s = int(sys.argv[1]); mode = sys.argv[2]; cmax = None if sys.argv[3] == 'none' else float(sys.argv[3])
    alphas = [float(x) for x in sys.argv[4].split(',')]; tries = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    path = os.path.join(ROOT, 'paper2', 'data', f'map_s{s}_{mode}_{sys.argv[3]}.jsonl')
    done = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line); done[(r['alpha'], r['L'])] = r
    log = open(path, 'a')
    for a in alphas:
        warm = None
        for L in levels(s, a, s + 1.5):
            key = (a, L)
            if key in done: r = done[key]
            else:
                t0 = time.time()
                r = try_level(s, a, L, mode, cmax, tries, warm)
                r.update(alpha=a, L=L, s=s, mode=mode, cmax=sys.argv[3], sec=round(time.time()-t0, 1))
                log.write(json.dumps(r) + '\n'); log.flush()
                print(json.dumps({k: v for k, v in r.items() if k != 'x'}), flush=True)
            if not r['found']:
                print(json.dumps(dict(alpha=a, s=s, mode=mode, cmax=sys.argv[3], q=L, summary=True)), flush=True)
                break
            warm = r['x']
