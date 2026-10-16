# fractional-vrk-barriers

Код и данные к статье Д. А. Приказчикова «Барьеры порядка явных дробных методов Рунге–Кутты в форме Вольтерры» (рукопись, 2026).

*Code and data for the paper "Order barriers for explicit fractional Runge–Kutta methods of Volterra type" (D. A. Prikazchikov, manuscript, 2026): an order-condition generator for explicit Volterra (Pouzet–Lubich) Runge–Kutta methods for Caputo fractional ODEs, barrier maps, optimal tableaux, convergence tests and an audit of the EFORK methods (Ghaffari et al., 2023).*

## Требования

Python ≥ 3.10, NumPy, SciPy, SymPy, Matplotlib (`pip install -r requirements.txt`). Все расчёты выполняются в двойной точности.

## Структура

```
src/        библиотека
  vrk_order.py    символьный генератор условий порядка (SymPy)
  vrk_fast.py     численный генератор: разложение локальной ошибки по h^(mα+k),
                  поиск методов минимизацией невязки (scipy.optimize.least_squares)
  core_fast.py    ядро условий в матричной форме (D_k, B_m, цепочка g2, вставки J')
  lincore.py      условия для линейной тестовой задачи
  vrk_global.py   глобальная схема: лаг-член по значениям стадий; таблицы 3A, 3B
  vrk_global2.py  лаг-член по узлам сетки (квадратичная интерполяция)
  vrk_global3.py  лаг-член по узлам сетки (степень r = 4), тестовые задачи (L), (N), (S)
scripts/    воспроизведение результатов статьи
data/       результаты расчётов (JSON / JSONL)
paper/      make_figs.py — построение рисунков по data/
```

## Что чем воспроизводится

| результат статьи | скрипт | данные |
|---|---|---|
| карта барьеров s = 1…3 | `scripts/vrk_map.py` | `data/map_s123.json` |
| карта барьеров s = 4 | `scripts/vrk_map4.py`, `vrk_map4b.py`, `vrk_map4c.py` | `data/map4*.json` |
| четырёхстадийные таблицы | `scripts/extract4.py`, `extract4b.py` | `data/tab4.json` |
| ядро условий, s = 3…6 | `src/core_fast.py` | — |
| пять стадий | `scripts/s5_high.py`, `s5_p.py`, `s5_cb.py` | `data/s5_scan.jsonl`, `data/s5_cb.jsonl` |
| патологическая ветвь s = 5 | `scripts/s5_cont.py` | `data/cont_*.log`, `data/s5_x_0.7_4.4_0.npy` |
| показатель локальной ошибки | `scripts/local.py` | `data/local.jsonl` |
| глобальная сходимость | `scripts/fullrecheck.py`, `collect_a07.py` | `data/full*.jsonl`, `data/data_a07.json` |
| лаг по стадиям и по сетке | `src/vrk_global.py`, `src/vrk_global2.py` | — |
| проверка EFORK | `scripts/audit.py`, `fork_test.py`, `check.py` | `data/g2.txt`, `data/g3.txt` |
| рисунки | `paper/make_figs.py` | `data/` |

Примеры:

```bash
python3 src/vrk_fast.py 0.7 3 3.0          # существует ли 3-стадийный метод с обнулёнными членами до h^3.0 при α = 0.7
python3 scripts/local.py 0.7               # показатели локальной ошибки при α = 0.7
python3 scripts/s5_cb.py 0.7 4.4 10 1.5    # поиск 5-стадийного метода, узлы c_i ≤ 1.5
python3 paper/make_figs.py                 # рисунки в paper/figs/
```

## Генератор условий порядка

Локальная ошибка одного шага при точном лаг-члене разлагается в ряд по двум переменным H₁ = h^α и H = h; коэффициенты — многочлены от формальных атомов g_k = (d/dt)^k f(t, y(t)) и D_{r,k} = (d/dt)^k ∂_y^r f(t, y(t)). Мономы H₁^m H^k с mα + k > p отбрасываются, мономы с одинаковым значением mα + k группируются (α приводится к рациональному виду `Fraction(α).limit_denominator(1000)`). Каждое ненулевое произведение атомов даёт одно условие. Существование метода проверяется методом Левенберга–Марквардта (`least_squares`, `xtol = ftol = gtol = 1e-15`, до 3000–4000 вычислений невязки, 7–60 случайных стартов); невязка ~1e-15 означает существование, устойчиво отделённая от нуля невязка (1e-6…1e-2) — численное свидетельство несуществования, но не доказательство.

## Лицензия

См. файл LICENSE.
