"""Generate LaTeX tables of order conditions (N+K<=6) and counts for paper 2."""
import os, sys, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src')]
from collections import defaultdict
from vtrees import trees_by_order, NK, sigma, diff_tex, weight_tex, atoms, height
def compact(s):
    s = re.sub(r"(?:J(?![\'^]))+", lambda m: 'J' if len(m.group()) == 1 else f'J^{{{len(m.group())}}}', s)
    s = re.sub(r"A{2,}", lambda m: f'A^{{{len(m.group())}}}', s)
    return s
def expo(N, K):
    a = '\\alpha' if N == 1 else f'{N}\\alpha'
    return f'{K}+{a}' if K else a
NKMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6
lev = defaultdict(list)
for n in range(1, 8):
    for t in trees_by_order(n):
        N, K = NK(t)
        if N + K <= NKMAX: lev[(N, K)].append(t)
rows = []
for (N, K) in sorted(lev, key=lambda x: (x[0] + x[1], -x[0])):
    ts = sorted(lev[(N, K)], key=lambda t: (height(t), weight_tex(t)))
    grp = {}; marks = 'abcdefgh'
    cnt = defaultdict(int)
    for t in ts: cnt[atoms(t)] += 1
    for t in ts:
        a = atoms(t)
        if cnt[a] > 1 and a not in grp: grp[a] = marks[len(grp)]
    first = True
    for t in ts:
        e = f'$h^{{{expo(N,K)}}}$' if first else ''
        g = f'$^{{{grp[atoms(t)]}}}$' if atoms(t) in grp else ''
        rows.append(f'{e} & ${compact(diff_tex(t))}${g} & ${compact(weight_tex(t))}$ & {sigma(t)} & {height(t)}\\\\')
        first = False
    rows.append('\\midrule')
rows[-1] = '\\bottomrule'
HEAD = r'''{\small
\begin{longtable}{lllcc}
\caption{Условия порядка до уровня $N+K=6$. $F(T)$~--- элементарный дифференциал (композиция справа налево, $f''_k=D_{2,k}$), $\Psi(T)$~--- вес, $\sigma$~--- коэффициент симметрии, ht~--- стадийная высота (при $s<\mathrm{ht}$ условие выполнено тождественно). Деревья, отмеченные одной буквой в пределах уровня, для скалярных уравнений дают одно условие.}\label{tab:conds}\\
\toprule
степень & $F(T)$ & $\Psi(T)$ & $\sigma$ & ht\\
\midrule
\endfirsthead
\toprule
степень & $F(T)$ & $\Psi(T)$ & $\sigma$ & ht\\
\midrule
\endhead
'''
open(os.path.join(ROOT, 'paper2', 'paper', 'tab_conditions.tex'), 'w').write(HEAD + '\n'.join(rows) + '\n\\end{longtable}\n}\n')
# counts table
cnt = defaultdict(int); sc = defaultdict(set)
for n in range(1, 12):
    for t in trees_by_order(n):
        N, K = NK(t); cnt[(N, K)] += 1; sc[(N, K)].add(atoms(t))
out = []
for N in range(1, 7):
    cells = []
    for K in range(0, 6):
        c = cnt.get((N, K), 0); cells.append('---' if c == 0 else f'{c}/{len(sc[(N,K)])}')
    out.append(f'{N} & ' + ' & '.join(cells) + '\\\\')
CH = r'''\begin{table}[ht]
\centering\small
\caption{Число условий порядка на уровне $h^{K+N\alpha}$: для систем / для скалярных уравнений.}\label{tab:counts}
\begin{tabular}{c|cccccc}
\toprule
$N\backslash K$ & 0 & 1 & 2 & 3 & 4 & 5\\
\midrule
'''
open(os.path.join(ROOT, 'paper2', 'paper', 'tab_counts.tex'), 'w').write(CH + '\n'.join(out) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

