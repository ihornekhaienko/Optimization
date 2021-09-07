import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------------
# Data definition

N = 17
eps = 0.01
delta = 0.001
k = 20
x_0 = [-1.2 * N, 1.5 * N]

F1 = lambda x: (x[0] - N) ** 2 + (2 * x[1] - N) ** 2
F2 = lambda x: 10 * (3 * N * x[1] - x[0] ** 2) ** 2 + (N - x[0]) ** 2

# ---------------------------------------------------------------------------------------------------------------------
# Methods

def dichotomy(F, x_k, idx, eps, delta, k, a, b):
    x = (a + b) / 2
    i = 0
    while abs(b - a) > eps and i < k:
        l = (a + b) / 2 - delta
        r = (a + b) / 2 + delta

        x_k[idx] = l
        tempA = F(x_k)
        x_k[idx] = r
        tempB = F(x_k)

        if tempA < tempB:
            b = r
        else:
            a = l
        x = (a + b) / 2

    return x

def coordinateDescent(F, x_k, eps, delta, k, a1, b1, a2, b2, x1List, x2List, fList):
    i = 0
    A = 0.0

    B = F(x_k)

    while abs(B - A) > eps and i < k:
        A = B
        for j in range(len(x_k)):
            if j == 0:
                x_k[j] = dichotomy(F, list.copy(x_k), j, eps, delta, k, a1, b1)
            else:
                x_k[j] = dichotomy(F, list.copy(x_k), j, eps, delta, k, a2, b2)
            x1List.append(x_k[0])
            x2List.append(x_k[1])
            fList.append(F(x_k))
        B = F(x_k)
        i += 1

    return x1List, x2List, fList

# ---------------------------------------------------------------------------------------------------------------------
# Tests

x_1_1, x_1_2, f_1 = coordinateDescent(F1, list.copy(x_0), eps, delta, k, -2 * N, 2 * N, -2 * N, 2 * N, [ x_0[0] ], [ x_0[1] ], [ F1(x_0) ])

x_2_1, x_2_2, f_2 = coordinateDescent(F2, list.copy(x_0), eps, delta, k, -2 * N, 2 * N, 0, 1.5 * N * N, [ x_0[0] ], [ x_0[1] ], [ F2(x_0) ])

# ---------------------------------------------------------------------------------------------------------------------
# Minimization graphs

X, Y = np.meshgrid(np.arange(-100, 100 ,0.1), np.arange(-3 * N,3 * N,0.1))

plt.title('F1(x1,x2)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.contour(X,Y,F1(np.array([X,Y])),np.arange(280, 3020, 280))
plt.plot(x_1_1, x_1_2, '-o')
plt.show()

plt.title('F2(x1,x2)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot(x_2_1, x_2_2, '-o')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Minimization tables

d1 = { 
     'x_1_1': x_1_1,
     'x_1_2': x_1_2,
      'f_1': f_1,
     }
dF1 = pd.DataFrame(data=d1)
print(dF1.to_string())

d2 = {
      'x_2_1': x_2_1,
      'x_2_2': x_2_2,
      'f_2': f_2,
     }
dF2 = pd.DataFrame(data=d2)
print(dF2.to_string())