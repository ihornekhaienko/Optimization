import numpy as np
import matplotlib.pyplot as plt
import pandas

# ---------------------------------------------------------------------------------------------------------------------
# Data definition

N = 17

eps = 0.0001
delta = 0.0001
tau = 0.618
k = 21

F1 = lambda x: (x - N) ** 2
F2 = lambda x: (x - N) ** 2 - N * x
F3 = lambda x: (x - N) ** 2 + N * (x ** 2)

a = -N
b = 3 * N

k_list = [i for i in range(k)]

f_opt_1 = [0.0 for i in range(k)]
f_opt_2 = [-361.25 for i in range(k)]
f_opt_3 = [272.944444444 for i in range(k)]

# ---------------------------------------------------------------------------------------------------------------------
# Method implementation

def GSS(F, a, b, eps, tau, k, f_min_k, x_min_k):
    x1 = a + tau * (b - a)
    x2 = b - tau * (b - a)
    for i in range(k):
        if F(x1) < F(x2):
            b = x2
        else:
            a = x1
        x1 = a + tau * (b - a)
        x2 = b - tau * (b - a)
        x = (a + b) / 2

        x_min_k.append(x)
        f_min_k.append(F(x))
    return f_min_k, x_min_k

# ---------------------------------------------------------------------------------------------------------------------
# Tests

f_g_1, x_g_1 = GSS(F1, a, b, eps, tau, k, [], [])
f_g_2, x_g_2 = GSS(F2, a, b, eps, tau, k, [], [])
f_g_3, x_g_3 = GSS(F3, a, b, eps, tau, k, [], [])

z_g_1 = getZ(f_opt_1, f_g_1, [])
z_g_2 = getZ(f_opt_2, f_g_2, [])
z_g_3 = getZ(f_opt_3, f_g_3, [])

# ---------------------------------------------------------------------------------------------------------------------
# Minimization graph

plt.title('x_min(k), golden section method')
plt.xlabel('k')
plt.ylabel('x_min(k)')
plt.scatter(k_list, x_g_1, label='F1(x)')
plt.scatter(k_list, x_g_2, label='F2(x)')
plt.scatter(k_list, x_g_3, label='F3(x)')
plt.legend()
plt.grid()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Minimization table

d = {'k': k_list,
      'x_g_1': x_g_1,
      'f_g_1': f_g_1,
      'x_g_2': x_g_2,
      'f_g_2': f_g_2,
      'x_g_3': x_g_3,
      'f_g_3': f_g_3
     }
dF1 = pandas.DataFrame(data=d1)
print(dF1.to_string())

# ---------------------------------------------------------------------------------------------------------------------
# z(k) graph

plt.title('z(k), golden section method')
plt.xlabel('k')
plt.ylabel('z(k)')
plt.scatter(k_list, z_g_1, label='F1(x)')
plt.scatter(k_list, z_g_2,  label='F2(x)')
plt.scatter(k_list, z_g_3, label='F3(x)')
plt.legend()
plt.grid()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# z(k) table

d2 = {'k': k_list,
      'z_g_1': z_g_1,
      'z_g_2': z_g_2,
      'z_g_3': z_g_3
     }
dF2 = pandas.DataFrame(data=d2)
print(dF2.to_string())