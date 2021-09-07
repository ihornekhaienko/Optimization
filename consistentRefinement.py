import numpy as np
import matplotlib.pyplot as plt
import pandas

# ---------------------------------------------------------------------------------------------------------------------
# Data definition

N = 17

f1 = lambda x: (x - N / 2) ** 2
f2 = lambda x: (x - N / 2) ** 2 + N * x
f3 = lambda x: (x - N / 2) ** 2 + N * x ** 2

n_1 = 400
n_2 = 200

x_min = -N
x_max = N

xlist = np.linspace(x_min, x_max, 1000)
get_x = lambda a, b, n: [a + i * (b - a) / n for i in np.arange(0, n)]

# ---------------------------------------------------------------------------------------------------------------------
# Methods

def bruteForce(f, x_min, x_max, n):
    m = x_min
    f_min = f(x_min)

    for i in range(n):
        x_i = x_min + i * ((x_max - x_min) / n)
        f_x_i = f(x_i)

        if f_min > f_x_i:
            m = x_i
            f_min = f_x_i

    return m


def consistentRefinement(f, x_min, x_max, n, k, min_list, k_list):
    m = x_min
    f_min = f(x_min)
    i_s = (x_min, x_max)

    for i in range(n):
        x_i = x_min + i * ((x_max - x_min) / n)
        f_x_i = f(x_i)

        if f_min > f_x_i:
            f_min = f_x_i
            m = x_i

            a = x_i - 1
            b = x_i + 1
            if a < x_min:
                a = -N
            if b > x_min:
                b = N

            i_s = (a - 1, b + 1)

    min_list.append(m)
    k_list.append(k)

    if k > 0:
        m, min_list, k_list = consistentRefinement(f, i_s[0], i_s[1], n, k - 1, min_list, k_list)

    return m, min_list, k_list


def analytic(f, x):
    f_v = np.vectorize(f)
    return min(f_v(x))


# ---------------------------------------------------------------------------------------------------------------------
# Tests

x1_m1 = bruteForce(f1, x_min, x_max, n_1)
x2_m1 = bruteForce(f2, x_min, x_max, n_1)
x3_m1 = bruteForce(f3, x_min, x_max, n_1)

x1_m2, min_list1, k_list1 = consistentRefinement(f1, x_min, x_max, n_2, n_2, [], [])
x2_m2, min_list2, k_list2 = consistentRefinement(f2, x_min, x_max, n_2, n_2, [], [])
x3_m2, min_list3, k_list3 = consistentRefinement(f3, x_min, x_max, n_2, n_2, [], [])

x1_a = analytic(f1, xlist)
x2_a = analytic(f2, xlist)
x3_a = analytic(f3, xlist)

min_list1 = np.array(min_list1)
min_list2 = np.array(min_list2)
min_list3 = np.array(min_list3)

# ---------------------------------------------------------------------------------------------------------------------
# Function graphs

plt.title('Function')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(xlist, f1(xlist), '-', label='f1(x)')
plt.plot(xlist, f2(xlist), '-', label='f2(x)')
plt.plot(xlist, f3(xlist), '-', label='f3(x)')
plt.legend()
plt.grid()
plt.show()

plt.title('x_min')
plt.xlabel('k')
plt.ylabel('x_min(k)')
plt.plot(k_list1, min_list1, '-', label='f1(x)')
plt.plot(k_list2, min_list2, '-', label='f2(x)')
plt.plot(k_list3, min_list3, '-', label='f3(x)')
plt.legend()
plt.grid()
plt.show()

plt.title('f(x_min)')
plt.xlabel('x_min')
plt.ylabel('f(x_min(k))')
plt.plot(min_list1, f1(min_list1), 'x', label='f1(x)')
plt.plot(min_list2, f2(min_list2), 'x', label='f2(x)')
plt.plot(min_list3, f3(min_list3), 'x', label='f3(x)')
plt.legend()
plt.grid()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Minimization tables

d1 = {'k1': k_list1,
      'x_min(k1)': min_list1,
      'k2': k_list2,
      'x_min(k2)': min_list2,
      'k3': k_list3,
      'x_min(k3)': min_list3,
      }
df1 = pandas.DataFrame(data=d1)
print(df1.to_string())

d2 = {'x_min(k1)': min_list1,
      'f(x_min(k1))': f1(min_list1),
      'x_min(k2)': min_list2,
      'f(x_min(k2))': f2(min_list2),
      'x_min(k3)': min_list3,
      'f(x_min(k3))': f3(min_list3),
     }
df2 = pandas.DataFrame(data=d2)
print(df2.to_string())

# ---------------------------------------------------------------------------------------------------------------------
# z(k) table

d = {'n': [1, 2, 3],
     'x_min 1': [x1_m1, x2_m1, x3_m1],
     'x_min 2': [x1_m2, x2_m2, x3_m2],
     'x_opt': [x1_a, x2_a, x3_a],
     'z_opt 1': [abs(x1_a - x1_m1), abs(x2_a - x2_m1), abs(x3_a - x3_m1)],
     'z_opt 2': [abs(x1_a - x1_m2), abs(x2_a - x2_m2), abs(x3_a - x3_m2)],
     }
df = pandas.DataFrame(data=d)
print(df.to_string())