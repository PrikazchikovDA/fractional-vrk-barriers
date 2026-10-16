import os, sys
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')]
from vtrees import all_trees, NK, height
def levels(s, a, top):
    return sorted({round(NK(t)[0]*a + NK(t)[1], 10) for t in all_trees(a, top) if height(t) <= s})
