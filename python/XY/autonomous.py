# import statements

import numpy as np
import numpy.random as ra
from tqdm import tqdm
import matplotlib.pyplot as plt


# Set N = side-length, F = number of frames, I = iterations per frame, L = last completed frame, and S = update density
N = 512
F = 3000
I = 200
L = -1
S = 0.05


# Define auxiliary functions (fast)

theta = None

def ZeroBC(i,j):
    if i >= 0 and i < N and j >= 0 and j < N:
        return theta[i,j]
    else:
        return 0

adjacency = [(0,1),(0,-1),(1,0),(-1,0)]

def Hloc(i,j,proptheta=None):
    if proptheta is None:
        proptheta = ZeroBC(i,j)
    return - sum([np.cos(proptheta - ZeroBC(i+a,j+b)) for (a,b) in adjacency])

def MetropolisUpdate(i,j,beta=1.1343):
    proptheta = ra.uniform(0,2*np.pi)
    if ra.uniform(0,1) < np.exp(- beta * (Hloc(i,j,proptheta) - Hloc(i,j))):
        theta[i,j] = proptheta


beta = lambda x : 1.1343 + 200 * (1+20*x**4) * (x-1/3)**5

# Run simulation and generate images (slow)

if L > -1:
    theta = np.load(f'data/{L:06d}.npy').reshape((N,N))
else:
    theta = ra.uniform(0,2*np.pi,size=(N,N))

for frame in tqdm(range(L+1, F)):
    for _ in tqdm(range(I), leave=False):
        to_update = ra.uniform(0,1,size=(N,N)) < 0.05
        for i in range(N):
            for j in range(N):
                if to_update[i,j]:
                    MetropolisUpdate(i,j,beta=beta(frame/F))

    np.save(f'data/{frame:06d}.npy', theta)
    plt.imsave(f'images/{frame:06d}.png', theta, format='png', cmap='hsv', vmin=0, vmax=2*np.pi)