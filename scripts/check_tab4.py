# Проверка коэффициентов четырёхстадийных методов (табл. «Четырёхстадийные методы» статьи).
# Коэффициенты из data/tab4.json подставляются в полный набор условий порядка,
# построенный генератором src/vrk_fast.py, — с полной точностью и после округления
# до шести знаков после запятой (как в таблице статьи).
#
#   python3 scripts/check_tab4.py
#
# Ожидаемый результат: невязка ~1e-16 при полной точности и 1e-7…1e-6 после округления.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'src')); DATA = _os.path.join(_ROOT, 'data')
import json, numpy as np
from vrk_fast import Setup

S_STAGES = 4
# (alpha, p): обнуляются все члены с показателями <= p, т.е. ниже q(4, alpha)
CASES = [(0.3, 2.5), (0.7, 3.7)]


def pack(c, A, b, s=S_STAGES):
    """Свободные параметры в порядке Setup.unpack: c_2..c_s, a_ij (j>=2), b."""
    v = list(c[1:])
    for i in range(1, s):
        for j in range(1, i):
            v.append(A[i, j])
    return np.array(v + list(b))


def main():
    tab = json.load(open(_os.path.join(DATA, 'tab4.json')))
    ok = True
    for a, p in CASES:
        d = tab[str(a)]
        c, A, b = np.array(d['c']), np.array(d['A']), np.array(d['b'])
        S = Setup(S_STAGES, a, p); keys = S.keys()
        # первый столбец A восстанавливается из условия на суммы строк — сверяем с файлом
        _, A_rec, _ = S.unpack(pack(c, A, b))
        rowsum_dev = np.abs(A_rec - A).max()
        res_full = np.abs(S.residual(pack(c, A, b), keys)).max()
        res_r6 = np.abs(S.residual(pack(np.round(c, 6), np.round(A, 6), np.round(b, 6)), keys)).max()
        print(f'alpha={a}: условий {len(keys)}, max|невязка| полная точность {res_full:.1e}, '
              f'округление 6 знаков {res_r6:.1e}, отклонение сумм строк {rowsum_dev:.1e}')
        ok &= res_full < 1e-12 and rowsum_dev < 1e-12
    print('OK' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    _sys.exit(main())
