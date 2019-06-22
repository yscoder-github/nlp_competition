import numpy as np
inp_len = np.array([22, 40, 38, 14, 26, 29, 38, 49, 25, 20, 26, 20, 20, 50, 19, 17])
print(inp_len)
sort_perm = np.array(sorted(range(len(inp_len)), key=lambda k: inp_len[k], reverse=True))
print(sort_perm)
sort_inp_len = inp_len[sort_perm]
sort_perm_inv = np.argsort(sort_perm)
print(sort_perm_inv)
print(inp_len[sort_perm_inv])

