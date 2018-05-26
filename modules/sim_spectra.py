import solve_sdos as sdos
import numpy as np

def sim_sdos(gamma, delta, T, Vmin, Vmax, offset=1.,slope=10.,amp=0.005,freq=2500,Vsampling=2000):
    V=np.linspace(Vmin, Vmax, Vsampling)
    E, C= sdos.get_mat(V,gamma,delta,T)
    simdos=offset-slope*E + amp*np.sin(freq*E)
    sim_dIdV=np.array(np.matrix(C)*np.transpose(np.matrix(simdos))).flatten()
    return [V,sim_dIdV],[E,simdos]