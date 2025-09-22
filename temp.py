val = 3.87 * 0.0232
count = 0
while val * 0.75 * 0.75 > 1e-4:
    count += 1
    val *= 0.75 * 0.75
    print(count, val)

from math import comb
n, _n = 22, 25
m = 15
print(f"{comb(n - 1, m - 1) / comb(n, m)} -> {comb(_n - 1, m - 1) / comb(_n, m)}")
