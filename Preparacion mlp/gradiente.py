import numpy as np
import matplotlib.pyplot as plt
import math as m

def gradienteDescendiente(f, fp, X, h, N):
    x_lim = np.linspace(-5, 5, 50)
    y_lim = np.linspace(-5, 5, 50)
    x, y = np.meshgrid(x_lim, y_lim)
    z = f(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    
    for i in range(N):
        X = X - h * fp(X[0], X[1])

    print(f"x = {X[0]}, y = {X[1]}")

    ax.scatter(X[0], X[1], f(X[0], X[1]), c='r', marker='*', s=100)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Gradiente Descendiente (3D)')

    plt.figure()
    plt.contour(x, y, z, 20)
    plt.scatter(X[0], X[1], c='r', marker='*', linewidths=2, label='óptimo')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradiente Descendiente (2D)')

    plt.show()

f = lambda x, y : 10 - np.exp(-(x**2 + 3*y**2))
fp = lambda x, y : np.array([2 * x * np.exp(-(x**2 + 3*y**2)), 6 * y * np.exp(-(x**2 + 3*y**2))])
X = np.array([1, 2])
h = 0.1
N = 100000
gradienteDescendiente(f, fp, X, h, N)