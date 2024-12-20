# import statements

import numpy as np
import numpy.random as ra
from tqdm import tqdm
import matplotlib.pyplot as plt
import colorcet as cc


# Set N = side-length, F = number of frames, I = iterations per frame, L = last completed frame, and S = update density
N = 512
F = 3000
I = 500
L = 1712
S = 0.05


# Define auxiliary functions (fast)

theta = None

def InsideInnerCircle(i,j):
    return (i+0.5-0.5*N)**2 + (j+0.5-0.5*N)**2 <= (0.5*N - 8)**2

def OutsideOuterCircle(i,j):
    return (i+0.5-0.5*N)**2 + (j+0.5-0.5*N)**2 > (0.5*N)**2

def CircleBC(i,j):
    if InsideInnerCircle(i,j):
        return theta[i,j]
    elif OutsideOuterCircle(i,j):
        return -10
    else:
        return np.arctan2(i+0.5-0.5*N, j+0.5-0.5*N)

adjacency = [(0,1),(0,-1),(1,0),(-1,0)]

def Hloc(i,j,proptheta):
    return - sum([np.cos(proptheta - theta[i+a,j+b]) for (a,b) in adjacency])

def MetropolisUpdate(i,j,beta=1.1343):
    proptheta = ra.uniform(-np.pi,np.pi)
    if ra.uniform(0,1) < np.exp(- beta * (Hloc(i,j,proptheta) - Hloc(i,j,theta[i,j]))):
        theta[i,j] = proptheta


beta = lambda x : 1.1343 + 150 * (1+50*x**8) * (x-1/3)**5

InsideCircleFast = np.array([[InsideInnerCircle(i,j) for i in range(N)] for j in range(N)])

cmap=cc.m_cyclic_rygcbmr_50_90_c64
cmap.set_under(color='black')

# Run simulation (slow)

if L > -1:
    theta = np.load(f'circledata/{L:06d}.npy').reshape((N,N))
else:
    theta = ra.uniform(-np.pi,np.pi,size=(N,N))

for i in range(N):
    for j in range(N):
        theta[i,j] = CircleBC(i,j)

for frame in tqdm(range(L+1, F)):
    b = beta(frame/F)

    for _ in tqdm(range(I), leave=False):
        to_update = np.logical_and(InsideCircleFast, ra.uniform(0,1,size=(N,N)) < 0.05)

        for i in range(N):
            for j in range(N):
                if to_update[i,j]:
                    MetropolisUpdate(i,j,beta=b)

    np.save(f'circledata/{frame:06d}.npy',theta)
    plt.imsave(f'circleimages/{frame:06d}.png', theta, format='png', cmap=cmap, vmin=-np.pi, vmax=np.pi)
