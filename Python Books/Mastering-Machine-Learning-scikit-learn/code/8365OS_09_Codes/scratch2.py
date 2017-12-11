import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r = 1
cx = []
cy = []
range = np.arange(0, 10, 0.4)
for i in range:
    cx.append(r * np.cos(i))
    cy.append(r * np.sin(i))


plt.scatter(cx, cy, marker='.', s=200)
plt.xticks([])
plt.yticks([])
bx = np.random.rand(10) - 0.5
by = np.random.rand(10) - 0.5
plt.scatter(bx, by, marker='x', s=200)
plt.show()

t_cx = []
t_cy = []
t_cz = []
for i, x in enumerate(cx):
    t_cx.append(x**2)
    t_cy.append(cy[i]**2)
    t_cz.append(math.sqrt(2) * x * cy[i])

t_bx = []
t_by = []
t_bz = []
for i, x in enumerate(bx):
    t_bx.append(x**2)
    t_by.append(by[i]**2)
    t_bz.append(math.sqrt(2) * x * by[i])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(t_bx, t_by, t_bz, marker='x')
ax.scatter(t_cx, t_cy, t_cz, marker='.')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.show()