import matplotlib.pyplot as plt

plt.axes().set_aspect('equal')
frame1 = plt.gca()
frame1.axes.get_xaxis().set_ticks([0, 1])
frame1.axes.get_yaxis().set_ticks([0, 1])
plt.ylim((-.2, 1.2))
plt.xlim((-.2, 1.2))
plt.scatter([1, 0, 1], [1, 1, 0], marker='.', c='k', s=800)
plt.scatter([0], [0],  marker='D', c='r', s=200)
plt.title('OR')
plt.plot([-.8, 1.8], [1.4, -1.4])
plt.show()