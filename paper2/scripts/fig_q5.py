"""Figure: five-stage barrier — old conjecture (paper 1), refined conjecture, and exponents of constructed methods."""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
a = np.linspace(0.02, 1, 400)
old = np.maximum(2+3*a, np.minimum(1+5*a, 3+2*a)); new = np.maximum(2+3*a, np.minimum(1+5*a, 4+a))
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(a, new, color='#1f4e79', lw=2, label=r'гипотеза: $\max(2+3\alpha,\min(1+5\alpha,4+\alpha))$')
ax.plot(a, old, color='#b03a2e', lw=1.2, ls='--', label=r'гипотеза работы [1]: $\max(2+3\alpha,\min(1+5\alpha,3+2\alpha))$')
low = [json.loads(l) for l in open(os.path.join(ROOT, 'paper2', 'data', 'branch_low.jsonl'))]
al = sorted({r['alpha'] for r in low if r['found']}); ax.plot(al, [2+3*x for x in al], 'o', ms=4, color='#444', label=r'построенные методы, $\alpha<1/2$')
A = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
ax.plot(A, [min(1+5*x, 4+x) for x in A], 's', ms=4, mfc='white', color='#1f4e79', label='ветвь A (изолированная 2-я стадия)')
ax.plot([1], [5], '*', ms=10, color='k', label=r'классика, $\alpha=1$: $q=5$')
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$q(5,\alpha)$'); ax.set_xlim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(ROOT, 'paper2', 'paper', 'figs', 'q5.pdf'))
