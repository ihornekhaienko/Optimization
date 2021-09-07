import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------------
# Data definition

F1 = lambda x: (x[0] - 3 * x[1]) ** 2 + (2 * x[1] - N) ** 2
F2 = lambda x: (N * x[1] - x[0] ** 2) ** 2 + (N - x[0]) ** 2

N = 17
x_0 = np.array([-N, -N])

# ---------------------------------------------------------------------------------------------------------------------
# Method implementation

def hooke_jeeves(function, x0, d, d_min, x1List, x2List, fList):
    n = x0.size
    e = np.eye(n) * d
    x = x0
    fx = function(x)
    num_iterations = 0

    while e[1, 1] > d_min:
        current_position = x
        for i in range(0, n):
            z = current_position + e[:, i]
            y = function(z)
            num_iterations += 1
            if y < fx:
                current_position = z
                fx = y
                if num_iterations % 100 == 0:
                    x1List.append(x[0])
                    x2List.append(x[1])
                    fList.append(fx)
                    
            else:
                z = current_position - e[:, i]
                y = function(z)
                num_iterations += 1
                if y < fx:
                    current_position = z
                    fx = y
                if num_iterations % 100 == 0:
                    x1List.append(x[0])
                    x2List.append(x[1])
                    fList.append(fx)


        if np.all(current_position == x):
            e = e * 0.5

        else:
            x1 = current_position + (current_position - x)
            f1 = function(x1)
            num_iterations += 1
            x = current_position
            if num_iterations % 100 == 0:
                x1List.append(x[0])
                x2List.append(x[1])
                fList.append(fx)

            if f1 < fx:
                x = x1
                fx = f1
                for i in range(0, n):
                    z = x1 - e[:, i]
                    y = function(z)
                    num_iterations += 1
                    if y < f1:
                        x = z
                        fx = y
                    if num_iterations % 100 == 0:
                        x1List.append(x[0])
                        x2List.append(x[1])
                        fList.append(fx)

    return x1List, x2List, fList

# ---------------------------------------------------------------------------------------------------------------------
# Tests

x_1_1, x_1_2, f_1 = hooke_jeeves(F1, x_0, 1, 0.0001, [x_0[0]], [x_0[1]], [F1(x_0)])
x_2_1, x_2_2, f_2 = hooke_jeeves(F2, x_0, 1, 0.0001, [x_0[0]], [x_0[1]], [F2(x_0)])

# ---------------------------------------------------------------------------------------------------------------------
# Minimization graphs

X, Y = np.meshgrid(np.arange(-2 * N,2 * N,0.1), np.arange(-2 * N,2 * N,0.1))

plt.title('F1(x1,x2)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.contour(X,Y,F1(np.array([X,Y])),np.arange(400, 4400, 400))
plt.plot(x_1_1, x_1_2, '-o')
plt.show()

plt.title('F2(x1,x2)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot(x_2_1[:30], x_2_2[:30], '-o')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Minimization tables

d1 = { 
     'x_1_1': x_1_1,
     'x_1_2': x_1_2,
      'f_1': f_1
     }
dF1 = pd.DataFrame(data=d1)
print(dF1.to_string())

d2 = { 
     'x_2_1': x_2_1[:30],
     'x_2_2': x_2_2[:30],
      'f_2': f_2[:30]
     }
dF2 = pd.DataFrame(data=d2)
print(dF2.to_string())