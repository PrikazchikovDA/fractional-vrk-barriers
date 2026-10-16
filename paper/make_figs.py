"""Figures for the paper (matplotlib, vector PDF)."""
import json, os, re, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gamma as G

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.join(HERE, '..', 'data')
FIG = os.path.join(HERE, 'figs'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    'font.family': 'Liberation Serif', 'mathtext.fontset': 'stix', 'font.size': 9,
    'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
    'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
    'legend.frameon': False, 'legend.fontsize': 8, 'savefig.bbox': 'tight'})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6']   # categorical slots, fixed order
MK = ['o', 's', '^', 'D', 'v']; LS = ['-', '--', '-.', ':', '-']
INK2 = '#52514e'

# ---------- Fig 1: EFORK consistency ratio and observed orders ----------
def tab2(a):
    g1, g2 = G(a+1), G(2*a+1); c2 = (2*g1**2/g2)**(1/a); return [0, c2], [1/(2*g1)]*2
def tab3(a, c2=0.5, c3=1.0):
    g1, g2, g3 = G(a+1), G(2*a+1), G(3*a+1); x2, x3 = c2**a, c3**a
    w = np.linalg.solve([[1, 1, 1], [0, x2, x3], [0, x2**2, x3**2]], [1/g1, g1/g2, g2/g3]); return [0, c2, c3], w
R = lambda c, w, a: sum(wi*ci**(1-a) for wi, ci in zip(w, c))/G(2-a)
al = np.linspace(0.15, 0.99, 120)
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
ax[0].plot(al, [R(*tab2(a), a) for a in al], color=C[0], ls=LS[0], label='2 стадии')
ax[0].plot(al, [R(*tab3(a), a) for a in al], color=C[1], ls=LS[1], label=r'3 стадии, $c=(0,\frac{1}{2},1)$')
ax[0].axhline(1, color=INK2, lw=0.8)
for a0 in (0.5, 1/3): ax[0].axvline(a0, color='#c3c2b7', lw=0.8, ls=':')
ax[0].set_ylim(0, 4); ax[0].set_xlabel(r'$\alpha$'); ax[0].set_ylabel(r'$R(\alpha)$')
ax[0].set_title('(а) коэффициент согласованности', fontsize=9); ax[0].legend(loc='upper right')
def parse(fn):
    out = {}
    for line in open(os.path.join(CODE, fn)):
        m = re.match(r'a=([\d.]+):', line)
        if m:
            e = [float(x) for x in re.findall(r'N=\d+:([\d.e+-]+)', line)]
            out[float(m.group(1))] = np.log(e[-3]/e[-2])/np.log(4)
    return out
for k, (fn, lab) in enumerate([('g2.txt', '2 стадии'), ('g3.txt', '3 стадии')]):
    d = parse(fn); xs = sorted(d)
    ax[1].plot(xs, [d[x] for x in xs], color=C[k], ls=LS[k], marker=MK[k], ms=4, label=lab + ' (наблюд.)')
ax[1].plot(al, 2*al, color=C[0], lw=0.8, alpha=0.6, label=r'заявлено $2\alpha$')
ax[1].plot(al, 3*al, color=C[1], lw=0.8, alpha=0.6, label=r'заявлено $3\alpha$')
ax[1].set_xlabel(r'$\alpha$'); ax[1].set_ylabel('наблюдаемый порядок'); ax[1].set_ylim(-0.4, 3)
ax[1].set_title(r'(б) EFORK, $N=160\to640$', fontsize=9); ax[1].legend(loc='upper left', fontsize=7)
fig.savefig(os.path.join(FIG, 'efork.pdf')); plt.close(fig)

# ---------- Fig 2: barrier map ----------
maps = []
for fn in ['map_s123.json', 'map4.json', 'map4b.json', 'map4c.json']: maps += json.load(open(os.path.join(CODE, fn)))
pts = {}
for r in maps:
    if r['err_exp'] is not None and r['s'] >= 2: pts.setdefault(r['s'], []).append((r['alpha'], r['err_exp']))
pts[5] = [(0.3, 2.9), (0.6, 4.0), (0.7, 4.4), (0.8, 4.6), (0.9, 4.8)]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
a_ = np.linspace(0.15, 1.0, 300)
form = {2: lambda a: 1+2*a, 3: lambda a: np.maximum(2+a, 1+3*a), 4: lambda a: np.maximum(2+2*a, 1+4*a),
        5: lambda a: np.maximum(2+3*a, np.minimum(1+5*a, 3+2*a))}
for k, s in enumerate([2, 3, 4, 5]):
    ax.plot(a_, form[s](a_), color=C[k], ls=LS[k], lw=1.4, label=f'$s={s}$' + (' (гипотеза 2)' if s == 5 else ''))
    xs, ys = zip(*sorted(pts[s])); ax.plot(xs, ys, ls='none', marker=MK[k], ms=5, color=C[k], mec='white', mew=0.8)
aw = np.linspace(2/3, 0.72, 20); ax.plot(aw, 1+5*aw, color=C[3], lw=0.9, ls=(0, (1, 1.5)), alpha=0.8)
ax.plot([1]*3, [3, 4, 5], ls='none', marker='*', ms=8, color=INK2, label=r'классика ($\alpha=1$)')
ax.axvline(0.5, color='#c3c2b7', lw=0.8, ls=':')
ax.text(0.51, 1.3, r'излом при $\alpha=\frac{1}{2}$', fontsize=7.5, color=INK2)
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$q(s,\alpha)$: локальная ошибка $O(h^{q})$')
ax.set_xlim(0.12, 1.03); ax.set_ylim(1.2, 6.2); ax.legend(loc='upper left')
fig.savefig(os.path.join(FIG, 'barriers.pdf')); plt.close(fig)

# ---------- Fig 3: convergence at alpha = 0.7 ----------
d = json.load(open(os.path.join(CODE, 'data_a07.json'))); Ns = np.array(d['Ns'])
titles = {'linear (Ex.1)': '(а) линейная задача', 'nonlinear y=t^4': '(б) нелинейное уравнение', '2D system': '(в) система 2×2'}
fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.5), sharey=False)
for ax, (p, errs) in zip(axs, d['err'].items()):
    for k, (m, e) in enumerate(errs.items()):
        q = d['q'][m]; e = np.array(e)
        ax.loglog(Ns, e, color=C[k], ls=LS[k], marker=MK[k], ms=4, label=f'{m}, $q={q:.1f}$')
        ax.loglog(Ns, e[0]*(Ns/Ns[0])**(-q), color=C[k], lw=0.7, alpha=0.45)
    ax.set_title(titles[p], fontsize=9); ax.set_xlabel('$N$'); ax.set_xticks(Ns); ax.set_xticklabels(Ns); ax.minorticks_off()
axs[0].set_ylabel(r'$\|y_N-y(1)\|_\infty$')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))
fig.savefig(os.path.join(FIG, 'convergence_a07.pdf')); plt.close(fig)

# ---------- Fig 4: stage-lag vs grid-lag ----------
A = [0.3, 0.5, 0.7, 0.9]
stage = {'2 стадии': [1.39, 1.55, 1.74, 1.93], '3 стадии, A': [1.45, 1.43, 1.65, 1.87]}
grid = {'2 стадии': [1.64, 2.03, 2.43, 2.83], '3 стадии, A': [2.32, 2.53, 2.71, 2.89]}
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for k, m in enumerate(stage):
    ax.plot(A, grid[m], color=C[k], ls='-', marker=MK[k], ms=5, label=f'{m}: лаг по сетке')
    ax.plot(A, stage[m], color=C[k], ls='--', marker=MK[k], ms=5, mfc='white', label=f'{m}: лаг по стадиям')
aa = np.linspace(0.25, 0.95, 50)
ax.plot(aa, 1+aa, color=INK2, lw=0.8, ls=':', label=r'$1+\alpha$')
ax.plot(aa, 1+2*aa, color=C[0], lw=0.7, alpha=0.5, label=r'$1+2\alpha$')
ax.plot(aa, 2+aa, color=C[1], lw=0.7, alpha=0.5, label=r'$2+\alpha$')
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('наблюдаемый глобальный порядок'); ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0))
fig.savefig(os.path.join(FIG, 'lag.pdf')); plt.close(fig)
print('ok')
