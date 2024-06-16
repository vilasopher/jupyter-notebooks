# import statements

import numpy as np
import numpy.random as ra
from tqdm import tqdm
import matplotlib.pyplot as plt
import colorcet as cc
import os
import multiprocessing as mp
from multiprocessing import shared_memory

# Set constants

N = 1000 # Side-length of square 
F = 3000 # Number of frames
I = 1000 # Number of iterations per frame
L = 1554 # Last completed frame
S = 0.1 # Proportion of resampled spins per pass
IRO = 20 # Inner radius offset (try to make N-2*IRO divisible by the number of cores)

# Inverse temperature, as a function of progress (from 0 to 1)
beta = lambda x : 1.1343 + 150 * (1+50*x**8) * (x-1/3)**5 

# Colormap
cmap=cc.m_cyclic_rygcbmr_50_90_c64
cmap.set_under(color='black')


# Define auxiliary functions

def InsideInnerCircle(i,j):
    return (i+0.5-0.5*N)**2 + (j+0.5-0.5*N)**2 <= (0.5*N-IRO)**2

def OutsideOuterCircle(i,j):
    return (i+0.5-0.5*N)**2 + (j+0.5-0.5*N)**2 > (0.5*N)**2

def CircleBC(theta,i,j):
    if InsideInnerCircle(i,j):
        return theta[i,j]
    elif OutsideOuterCircle(i,j):
        return -10
    else:
        return np.arctan2(i+0.5-0.5*N, j+0.5-0.5*N)

adjacency = [(0,1),(0,-1),(1,0),(-1,0)]

def Hloc(theta,i,j,proptheta):
    return - sum([np.cos(proptheta - theta[i+a,j+b]) for (a,b) in adjacency])

def MetropolisUpdate(theta,i,j,beta=1.1343):
    proptheta = ra.uniform(-np.pi,np.pi)
    if ra.uniform(0,1) < np.exp(- beta * (Hloc(theta,i,j,proptheta) - Hloc(theta,i,j,theta[i,j]))):
        theta[i,j] = proptheta

def worker(core, shm_name, start, end, beta):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_theta = np.ndarray((N,N), dtype=np.float64, buffer=existing_shm.buf)

    for iteration in tqdm(range(I), position=1, leave=False) if core == 0 else range(I):
        for i in range(start,end) if iteration % 2 == 0 else range(end-1,start-1,-1):
            for j in range(N) if iteration % 4 < 2 else range(N-1,-1,-1):
                if InsideInnerCircle(i,j) and ra.uniform(0,1) < S:
                    MetropolisUpdate(shared_theta,i,j,beta)
    
    existing_shm.close()


# Run simulation (slow)

def main():

    if L > -1:
        theta = np.load(f'paralleldata/{L:06d}.npy').reshape((N,N))
    else:
        theta = ra.uniform(-np.pi,np.pi,size=(N,N))

    for i in range(N):
        for j in range(N):
            theta[i,j] = CircleBC(theta,i,j)

    num_cores = os.cpu_count()
    shm = shared_memory.SharedMemory(create=True, size=theta.nbytes)
    shared_theta = np.ndarray(theta.shape, dtype=np.float64, buffer=shm.buf)
    np.copyto(shared_theta, theta)

    chunk_size = (N-2*IRO) // num_cores

    for frame in tqdm(range(L+1, F)):
        b = beta(frame/F)
        processes = []

        for c in range(num_cores):
            start = IRO + c * chunk_size
            end = IRO + (c+1) * chunk_size if c < num_cores-1 else N-IRO
            p = mp.Process(target=worker, args=(c, shm.name, start, end, b))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        #print('\033[12A', end='')
        print('Saving data... Please do not turn the power off...', end='')
        np.copyto(theta, shared_theta)
        np.save(f'paralleldata/{frame:06d}.npy',theta)
        plt.imsave(f'parallelimages/{frame:06d}.png', theta, format='png', cmap=cmap, vmin=-np.pi, vmax=np.pi)
        print('\033[1A', end='')
    
    shm.close()
    shm.unlink()


if __name__ == '__main__':
    main()