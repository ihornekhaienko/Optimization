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

def dichotomy(F, a, b, eps, delta, k, f_min_k, x_min_k):
    x = (a + b)/ 2.0
    l = x - delta
    r = x + delta

    for i in range(k):
        if F(l) <= F(r):
            b = r
        else:
            a = l
        x = (a + b)/ 2.0
        l = x - delta
        r = x + delta

        x_min_k.append(x)
        f_min_k.append(F(x))

    return f_min_k, x_min_k

def getZ(x_opt, x_k, z):
    for i in range(len(x_k)):
        z.append(abs(x_opt[i] - x_k[i]))
    return z

# ---------------------------------------------------------------------------------------------------------------------
# Tests

f_d_1, x_d_1 = dichotomy(F1, a, b, eps, delta, k, [], [])
f_d_2, x_d_2 = dichotomy(F2, a, b, eps, delta, k, [], [])
f_d_3, x_d_3 = dichotomy(F3, a, b, eps, delta, k, [], [])

z_d_1 = getZ(f_opt_1, f_d_1, [])
z_d_2 = getZ(f_opt_2, f_d_2, [])
z_d_3 = getZ(f_opt_3, f_d_3, [])

# ---------------------------------------------------------------------------------------------------------------------
# Minimization graph

plt.title('x_min(k), dichotomy method')
plt.xlabel('k')
plt.ylabel('x_min(k)')
plt.scatter(k_list, x_d_1, label='F1(x)')
plt.scatter(k_list, x_d_2,  label='F2(x)')
plt.scatter(k_list, x_d_3, label='F3(x)')
plt.legend()
plt.grid()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Minimization table

d1 = {'k': k_list,
      'x_d_1': x_d_1,
      'f_d_1': f_d_1,
      'x_d_2': x_d_2,
      'f_d_2': f_d_2,
      'x_d_3': x_d_3,
      'f_d_3': f_d_3
     }
dF1 = pandas.DataFrame(data=d1)
print(dF1.to_string())

# ---------------------------------------------------------------------------------------------------------------------
# z(k) graph

plt.title('z(k), dichotomy method')
plt.xlabel('k')
plt.ylabel('z(k)')
plt.scatter(k_list, z_d_1, label='F1(x)')
plt.scatter(k_list, z_d_2,  label='F2(x)')
plt.scatter(k_list, z_d_3, label='F3(x)')
plt.legend()
plt.grid()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# z(k) table

d2 = {'k': k_list,
      'z_d_1': z_d_1,
      'z_d_2': z_d_2,
      'z_d_3': z_d_3
     }
dF2 = pandas.DataFrame(data=d2)
print(dF2.to_string())