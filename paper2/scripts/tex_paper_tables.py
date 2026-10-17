"""LaTeX tables for paper 2: Butcher tableaux of branch A (from 50-digit polished data) and the S-conditions table."""
import os, sys, json, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src'), os.path.join(ROOT, 'paper2', 'scripts')]
import mpmath as mp
from polish import build
from vtrees import trees_by_order, NK, height, sigma, diff_tex, weight_tex
mp.mp.dps = 50
recs = {}
for l in open(os.path.join(ROOT, 'paper2', 'data', 'branchA_verify.jsonl')):
    r = json.loads(l); recs[r['alpha']] = r
def f(x): 
    x = float(x); return '0' if abs(x) < 5e-13 else f'{x:.6f}'
blocks = []
for a in [0.6, 0.7, 0.8, 0.9]:
    vm = [mp.mpf(x) for x in recs[a]['v']]; c, A, b = build(vm, 5, mp.mpf(str(a)))
    rows = [f'{f(c[i])} & ' + ' & '.join(f(A[i][j]) if j < i else '' for j in range(5)) + r'\\' for i in range(5)]
    q = min(1 + 5*a, 4 + a)
    blocks.append(rf'''\begin{{minipage}}[t]{{0.49\textwidth}}\centering
$\alpha={a}$, $q={q:g}$\\[2pt]
{{\scriptsize\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{r|rrrrr}}
{chr(10).join(rows)}
\hline
 & {' & '.join(f(x) for x in b)}
\end{{tabular}}}}
\end{{minipage}}''')
tex = r'''\begin{table}[ht]
\centering
\caption{Пятистадийные методы ветви A (класс с изолированной второй стадией) в виде таблиц Бутчера $c\,|\,A$ над $b^{\top}$; шесть знаков после запятой. Полная (50-значная) точность~--- файл \texttt{paper2/data/branchA\_verify.jsonl} репозитория.}\label{tab:butcher5}
''' + blocks[0] + '\\hfill\n' + blocks[1] + '\\\\[8pt]\n' + blocks[2] + '\\hfill\n' + blocks[3] + '\n\\end{table}\n'
open(os.path.join(ROOT, 'paper2', 'paper', 'tab_butcher5.tex'), 'w').write(tex)
# S table
def compact(s):
    s = re.sub(r"(?:J(?![\\'^]))+", lambda m: 'J' if len(m.group()) == 1 else f'J^{{{len(m.group())}}}', s)
    return re.sub(r"A{2,}", lambda m: f'A^{{{len(m.group())}}}', s)
def expo(N, K):
    a = r'\alpha' if N == 1 else f'{N}\\alpha'
    return f'{K}+{a}' if K else a
S = [t for n in range(1, 7) for t in trees_by_order(n) if NK(t)[1] <= 3 and sum(NK(t)) <= 5 and height(t) <= 5]
S.sort(key=lambda t: (sum(NK(t)), NK(t)[1], weight_tex(t)))
rows = []
for t in S:
    N, K = NK(t); d1 = 'да' if 'd_{1}' in weight_tex(t) else ''
    rows.append(f'$h^{{{expo(N,K)}}}$ & ${compact(diff_tex(t))}$ & ${compact(weight_tex(t))}$ & {sigma(t)} & {d1}\\\\')
tex = r'''\begin{table}[ht]
\centering\small
\caption{Набор $\mathcal S$: шестнадцать условий, обнуление которых при $\alpha>\frac23$ равносильно превышению показателя $3+2\alpha$ пятистадийным методом. Последний столбец отмечает деревья с листом $d_1$, которые зануляются упрощающими условиями~\eqref{eq:simpl}.}\label{tab:S}
\begin{tabular}{lllcc}
\toprule
степень & $F(T)$ & $\Psi(T)$ & $\sigma$ & лист $d_1$\\
\midrule
''' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
open(os.path.join(ROOT, 'paper2', 'paper', 'tab_S.tex'), 'w').write(tex)
print(len(S), 'S trees')
