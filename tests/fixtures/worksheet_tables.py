"""REBA and RULA lookup tables — ground truth for tests/test_tables.py.

Transcribed cell by cell on 2026-09-17 from the ErgoPlus worksheets (tables rendered at 4x):
  REBA: https://ergo-plus.com/wp-content/uploads/REBA.pdf  (Hignett & McAtamney, Applied Ergonomics 31 (2000) 201-205)
  RULA: https://ergo-plus.com/wp-content/uploads/RULA.pdf  (McAtamney & Corlett, Applied Ergonomics 24(2) (1993) 91-99)
and verified programmatically, every cell, against an independent source each:
  REBA: github.com/rs9000/ergonomics ergonomics/reba.py (table_a, table_b, table_c) — 0 mismatches
  RULA: ieh.sg/tool/rula (Institute of Ergonomics and Hygiene calculator JS tableA/B/C) — 0 mismatches
(github.com/sksnnjj/rula-app was also checked and REJECTED: its Table A and B disagree with both sources.)

Production code must never import this file.
"""
REBA_A = {  # [neck][trunk][legs]
 1: [[1,2,3,4],[2,3,4,5],[2,4,5,6],[3,5,6,7],[4,6,7,8]],
 2: [[1,2,3,4],[3,4,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9]],
 3: [[3,3,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9],[7,8,9,9]],
}
REBA_B = {  # [upper_arm][lower_arm][wrist]
 1: [[1,2,2],[1,2,3]], 2: [[1,2,3],[2,3,4]], 3: [[3,4,5],[4,5,5]],
 4: [[4,5,5],[5,6,7]], 5: [[6,7,8],[7,8,8]], 6: [[7,8,8],[8,9,9]],
}
REBA_C = [
 [1,1,1,2,3,3,4,5,6,7,7,7],[1,2,2,3,4,4,5,6,6,7,7,8],[2,3,3,3,4,5,6,7,7,8,8,8],
 [3,4,4,4,5,6,7,8,8,9,9,9],[4,4,4,5,6,7,8,8,9,9,9,9],[6,6,6,7,8,8,9,9,10,10,10,10],
 [7,7,7,8,9,9,9,10,10,11,11,11],[8,8,8,9,10,10,10,10,10,11,11,11],[9,9,9,10,10,10,11,11,11,12,12,12],
 [10,10,10,11,11,11,11,12,12,12,12,12],[11,11,11,11,12,12,12,12,12,12,12,12],[12]*12,
]
RULA_A = {  # [upper_arm][lower_arm] -> [w1t1,w1t2,w2t1,w2t2,w3t1,w3t2,w4t1,w4t2]
 1: [[1,2,2,2,2,3,3,3],[2,2,2,2,3,3,3,3],[2,3,3,3,3,3,4,4]],
 2: [[2,3,3,3,3,4,4,4],[3,3,3,3,3,4,4,4],[3,4,4,4,4,4,5,5]],
 3: [[3,3,4,4,4,4,5,5],[3,4,4,4,4,4,5,5],[4,4,4,4,4,5,5,5]],
 4: [[4,4,4,4,4,5,5,5],[4,4,4,4,4,5,5,5],[4,4,4,5,5,5,6,6]],
 5: [[5,5,5,5,5,6,6,7],[5,6,6,6,6,7,7,7],[6,6,6,7,7,7,7,8]],
 6: [[7,7,7,7,7,8,8,9],[8,8,8,8,8,9,9,9],[9,9,9,9,9,9,9,9]],
}
RULA_B = {  # [neck] -> [t1l1,t1l2,...,t6l1,t6l2]
 1: [1,3,2,3,3,4,5,5,6,6,7,7], 2: [2,3,2,3,4,5,5,5,6,7,7,7], 3: [3,3,3,4,4,5,5,6,6,7,7,7],
 4: [5,5,5,6,6,7,7,7,7,7,8,8], 5: [7,7,7,7,7,8,8,8,8,8,8,8], 6: [8,8,8,8,8,8,8,9,9,9,9,9],
}
RULA_C = [[1,2,3,3,4,5,5],[2,2,3,4,4,5,5],[3,3,3,4,4,5,6],[3,3,3,4,5,6,6],
          [4,4,4,5,6,7,7],[4,4,5,6,6,7,7],[5,5,6,6,7,7,7],[5,5,6,7,7,7,7]]
