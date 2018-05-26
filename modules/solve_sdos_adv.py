import numpy as np
import solve_sdos as slv_sdos
from scipy.optimize import curve_fit

def get_rec_dIdV(V, gamma, delta, T, dIdV, E_sampling=3000, rcond=1e-5):
    sdos, C=slv_sdos.get_sdos(V, dIdV, gamma, delta, T,E_sampling=E_sampling,rcond=rcond, return_matrix=True)
    recon_dIdV=np.array(C*np.transpose(np.matrix(sdos))).flatten()
    return recon_dIdV


def get_params(V,dIdV,guess,lb,ub,E_sampling=3000,rcond=1e-5):
    fn = lambda x,gamma,delta,T: get_rec_dIdV(x,gamma,delta,T,dIdV=dIdV,E_sampling=E_sampling,rcond=rcond)
    par,par_cov=curve_fit(fn,V,dIdV,p0=guess,bounds=[lb,ub])
    E,sdos=slv_sdos.get_sdos(V, dIdV, par[0], par[1], par[2],E_sampling=E_sampling,rcond=rcond,return_matrix=False)
    return E,sdos,par