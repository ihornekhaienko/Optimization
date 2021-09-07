import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------------
# Data definition

N = 17
#N = 1
a = -2 * N
b = 2 * N

x_0_1 = [1.2 * N, 1.5 * N]
x_0_2 = [-1.2 * N, -1.5 * N]

F1 = lambda x: (x[0] - N/2) ** 2 + (2 * x[1] - N) ** 2
F2 = lambda x: 10 * (N * x[1] - x[0] ** 2) ** 2 + (N - x[0]) ** 2

phi = (1.0 + np.sqrt(5.0))/2.0

k_max = 20
eps = 0.001


# ---------------------------------------------------------------------------------------------------------------------
# Methods

def GSS(F, X, d, prev_val, lower, upper, eps):
    
    x1 = upper - ((phi - 1)*(upper - lower))
    x2 = lower + ((phi - 1)*(upper - lower))
    val = x1
    
    param2 = X - np.dot(x2, d)
    param2 = param2.tolist()
    
    param1 = X - np.dot(x1, d)
    param1 = param1.tolist()
    
    if F(param2) < F(param1):
        if x1 > x2:
            upper = x1
        else:
            lower = x1

    else:
        if x2 > x1:
            upper = x2
        else:
            lower = x2

    if abs(prev_val - val) <= eps:
        return val
    else:
        return GSS(F, X, d, val, lower, upper, eps)

def grad(f, X):

    h = 0.0000001
    delf = []
    
    for i in range(len(X)):
        E = np.zeros(len(X))
        E[i] = h
        vals = X + E
        delf.append((f(vals) - f(X))/h)
            
    return delf

def difference(X, Y):

    total = 0
    
    for i in range(len(X)):
        total = total + abs(X[i] - Y[i])
    total = total / len(X)

    return total

def gradientDescent(F, x_0, eps, k_max, a, b, x1List, x2List, fList):
    
    x_k = x_0
    
    for i in range(k_max + 1):
        d_k = grad(F, x_k)
        x_prev = x_k

        lym_k = GSS(F, x_k, d_k, 1, a, b, 0.001)
        x_k = x_k - np.dot(lym_k, d_k)
        x_k = x_k.tolist()

        x1List.append(x_prev[0])
        x2List.append(x_prev[1])
        fList.append(F(x_prev))

        if difference(x_prev, x_k) < eps:
            return x1List, x2List, fList
        
    return x1List, x2List, fList

# ---------------------------------------------------------------------------------------------------------------------
# Tests

x_1_1, x_1_2, f_1 = gradientDescent(F1, x_0_1, eps, k_max, a, b, [], [], [])
x_2_1, x_2_2, f_2 = gradientDescent(F1, x_0_2, eps, k_max, a, b, [], [], [])
#x_1_1, x_1_2, f_1 = gradientDescent(F2, x_0_1, eps, k_max, a, b, [], [], [])
#x_2_1, x_2_2, f_2 = gradientDescent(F2, x_0_2, eps, k_max, a, b, [], [], [])


# ---------------------------------------------------------------------------------------------------------------------
# Minimization graphs

X, Y = np.meshgrid(np.arange(-2 * N,2 * N,0.1), np.arange(-2* N,2 * N,0.1))

plt.title('F(x1,x2)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.contour(X,Y,F1(np.array([X,Y])),np.arange(50, 1600, 150))
#plt.contour(X,Y,F2(np.array([X,Y])),np.arange(1, 11, 1))
plt.plot(x_1_1, x_1_2, '-o', label = 'P1')
plt.plot(x_2_1, x_2_2, '-o', label = 'P2')
plt.legend()
plt.show()

plt.title('F(P1)')
plt.xlabel('k')
plt.ylabel('F')
plt.plot([i for i in range(len(f_1))], f_1, '-o')
plt.show()

plt.title('F(P2)')
plt.xlabel('k')
plt.ylabel('F')
plt.plot([i for i in range(len(f_2))], f_2, '-o')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Minimization tables

d1 = { 
     'f_1': f_1,
     'x_1_1': x_1_1,
     'x_1_2': x_1_2,
     }
dF1 = pd.DataFrame(data=d1)
print(dF1.to_string())

d2 = { 
     'f_2': f_2,
     'x_2_1': x_2_1,
     'x_2_2': x_2_2,
     }
dF2 = pd.DataFrame(data=d2)
print(dF2.to_string())