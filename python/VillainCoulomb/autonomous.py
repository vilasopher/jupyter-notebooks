import numpy as np
import numpy.random as ra
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
from datetime import timedelta

W = 64
H = 64

num_frames = 3000
iterations_per_frame = 1024
num_iterations = num_frames * iterations_per_frame

eee = np.exp(np.exp(np.e-1)-1)-1
def beta_function(iteration):
    return np.log(np.log(np.log(iteration * eee / num_iterations + 1) + 1) + 1)

vertices = [(vr, vc) for vr in range(H) for vc in range(W)]
numv = len(vertices)

edges = [(e1r, e1c, e2r, e2c) for e1r in range(H) for e1c in range(W) for (e2r, e2c) in [(e1r, e1c+1), (e1r+1, e1c)] if e2r < H and e2c < W]
nume = len(edges)

plaquettes = [(pr, pc) for pr in range(H-1) for pc in range(W-1)]
nump = len(plaquettes)

# Load operators from disk instead of generating them

d0 = np.load(f'data/{W}x{H}/matrices/d0.npy')
d1 = np.load(f'data/{W}x{H}/matrices/d1.npy')
dstar1 = np.load(f'data/{W}x{H}/matrices/dstar1.npy')
dstar2 = np.load(f'data/{W}x{H}/matrices/dstar2.npy')
L0 = np.load(f'data/{W}x{H}/matrices/L0.npy')
L1 = np.load(f'data/{W}x{H}/matrices/L1.npy')
L2 = np.load(f'data/{W}x{H}/matrices/L2.npy')

print('starting clock...')
starttime = time.time()

# generate all data

beta = beta_function(1)

theta = 2*np.pi*ra.random(numv)
m = np.round(ra.normal(- d0 @ theta / (2*np.pi), 1/(2*np.pi*np.sqrt(beta))))

def Hamiltonian(t,m):
    return np.sum((d0 @ t + 2*np.pi*m)**2)/2

Hold = Hamiltonian(theta, m)
Hnew = 0

ad = np.abs(d0)

for round in range(num_iterations):

    beta = beta_function(round+1)
    
    changedvertices = np.zeros(numv)
    changedvertices[ra.randint(numv)] = 1
    fixedvertices = 1 - changedvertices
    changededges = ad @ changedvertices
    changededges[changededges != 0] = 1
    fixededges = 1 - changededges

    proptheta = theta * fixedvertices + 2*np.pi*ra.random(numv) * changedvertices
    propm = m * fixededges + np.round(ra.normal(-d0 @ proptheta / (2*np.pi), 1/(2*np.pi*np.sqrt(beta)))) * changededges

    Hnew = Hamiltonian(proptheta, propm)

    if ra.random() < np.exp(- beta * (Hnew - Hold)):
        theta = proptheta
        m = propm
        Hold = Hnew
    
    if (round+1) % iterations_per_frame == 0:
        q = d1 @ m
        nq = la.lstsq(d1, q, rcond=None)[0]
        psi = la.lstsq(d0, m-nq, rcond=None)[0]
        phi = theta + 2*np.pi*psi + 2*np.pi*(dstar1 @ la.lstsq(L1, nq, rcond=None)[0])
        np.save(f'data/{W}x{H}/theta/{round//iterations_per_frame:06d}', theta)
        np.save(f'data/{W}x{H}/m/{round//iterations_per_frame:06d}', m)
        np.save(f'data/{W}x{H}/q/{round//iterations_per_frame:06d}', q)
        np.save(f'data/{W}x{H}/phi/{round//iterations_per_frame:06d}', phi)

        print(f'finished frame {round//iterations_per_frame:06d} in time {str(timedelta(seconds=time.time()-starttime))}')

thetas = [np.load(f'data/{W}x{H}/theta/{i:06d}.npy').reshape((H,W)) for i in range(num_frames)]
ms = [np.load(f'data/{W}x{H}/m/{i:06d}.npy') for i in range(num_frames)]
qs = [np.load(f'data/{W}x{H}/q/{i:06d}.npy').reshape((H-1,W-1)) for i in range(num_frames)]
phis = [np.load(f'data/{W}x{H}/phi/{i:06d}.npy').reshape((H,W)) for i in range(num_frames)]


for i in range(num_iterations//iterations_per_frame):
    fig, (tviewer, pviewer) = plt.subplots(1, 2, figsize=(16,8))
    tviewer.axes.get_yaxis().set_visible(False)
    tviewer.axes.get_xaxis().set_visible(False)
    pviewer.axes.get_yaxis().set_visible(False)
    pviewer.axes.get_xaxis().set_visible(False)

    tv = tviewer.matshow(thetas[i], vmin=0, vmax=2*np.pi, cmap='hsv')
    tviewer.set_title('Villain spins with wrap-around 1-form')
    fig.colorbar(tv, ax=tviewer, location='left', pad=0.05)

    mdown =  [(e1c, e1r+0.5, ms[i][j]) for (j, (e1r, e1c, e2r, e2c)) in enumerate(edges) if e2r > e1r and ms[i][j] > 0]
    mup =  [(e1c, e1r+0.5, ms[i][j]) for (j, (e1r, e1c, e2r, e2c)) in enumerate(edges) if e2r > e1r and ms[i][j] < 0]
    mright =  [(e1c+0.5, e1r, ms[i][j]) for (j, (e1r, e1c, e2r, e2c)) in enumerate(edges) if e2c > e1c and ms[i][j] > 0]
    mleft =  [(e1c+0.5, e1r, ms[i][j]) for (j, (e1r, e1c, e2r, e2c)) in enumerate(edges) if e2c > e1c and ms[i][j] < 0]

    markers = [[(-1,0),(1,0)], [(-1,0),(1,0)], [(0,-1),(0,1)], [(0,-1),(0,1)]]

    for (d,mlist) in enumerate([mdown, mup, mright, mleft]):
        mx = [x - 0.01 for (x,y,c) in mlist]
        my = [y - 0.01 for (x,y,c) in mlist]
        mc = [np.abs(c) for (x,y,c) in mlist]
        tviewer.scatter(mx, my, c=mc, marker=markers[d], linewidth=1, s=50, vmin=1, vmax=2.5, cmap='bone')

    pv = pviewer.matshow(phis[i], vmin=0, vmax=2*np.pi, cmap='PiYG')
    pviewer.set_title('Gaussian spin wave with Coulomb gas')
    fig.colorbar(pv, ax=pviewer, location='right')

    qpoints = [(pc+0.5, pr+0.5, qs[i][pr,pc]) for (pr, pc) in plaquettes if qs[i][pr,pc] != 0]
    qx = [x for (x,y,c) in qpoints]
    qy = [y for (x,y,c) in qpoints]
    qc = [c for (x,y,c) in qpoints]
    qv1 = pviewer.scatter(qx, qy, c=qc, marker='o', s=25, vmin = -2, vmax = 2, cmap='seismic')
    qv2 = tviewer.scatter(qx, qy, c=qc, marker='o', s=25, vmin = -2, vmax = 2, cmap='seismic', edgecolors='black')

    fig.text(0.4, 0.05, f'temperature = {1/beta_function((i+1)*iterations_per_frame):.5f}')
    plt.tight_layout()
    plt.savefig(f'images/{W}x{H}/{i:06d}.png', bbox_inches='tight')
    plt.close('all')