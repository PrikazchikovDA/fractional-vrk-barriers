"""Branch 2+3alpha for alpha<1/2 (systems, nodes in [0,1.5]): zero all trees with |T| < 2+3alpha."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src')]
from search import search_lm
log = open(os.path.join(ROOT, 'paper2', 'data', 'branch_low.jsonl'), 'a')
for a in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.48]:
    r = search_lm(5, a, 2 + 3*a, 'system', 1.5, tries=60, seed=7, inclusive=False)
    r.update(alpha=a, target=2+3*a); log.write(json.dumps(r) + '\n'); log.flush()
    print(json.dumps({k: v for k, v in r.items() if k != 'x'}), flush=True)
