import numpy as np
import solve_sdos as slvsdos
from scipy.optimize import curve_fit
from scipy.sparse.linalg import lsmr as lsmr


def get_rec_dIdV(V, gamma, delta, T, dIdV,atol=1e-3,btol=1e-3, E_sampling=300):
    E,C=slvsdos.get_mat(V,gamma, delta, T,E_sampling=E_sampling)
    sdos=lsmr(C,dIdV,atol=atol,btol=btol)[0]
    recon_dIdV=np.array(C*np.transpose(np.matrix(sdos))).flatten()
    return recon_dIdV



def get_params(V,dIdV,guess,lb,ub,err=[],atol=1e-3,btol=1e-3,E_sampling=300):

    fn = lambda x,gamma,delta,T: get_rec_dIdV(x,gamma,delta,T,dIdV=dIdV,atol=atol,btol=btol,E_sampling=E_sampling)

    if np.size(err)==0:
        err=dIdV*0.001
        err[abs(V)>0.003]=1.

    par,par_cov=curve_fit(fn,V,dIdV,p0=guess,sigma=err,bounds=[lb,ub])
    return par


#Old way of recovering sample DOS:
#def get_rec_dIdV(V, gamma, delta, T, dIdV, E_sampling=3000, rcond=1e-5):
#    sdos, C=slv_sdos.get_sdos(V, dIdV, gamma, delta, T,E_sampling=E_sampling,rcond=rcond, return_matrix=True)
#    recon_dIdV=np.array(C*np.transpose(np.matrix(sdos))).flatten()
#    return recon_dIdV
#
#
#def get_params(V,dIdV,guess,lb,ub,E_sampling=3000,rcond=1e-5):
#    fn = lambda x,gamma,delta,T: get_rec_dIdV(x,gamma,delta,T,dIdV=dIdV,E_sampling=E_sampling,rcond=rcond)
#    par,par_cov=curve_fit(fn,V,dIdV,p0=guess,bounds=[lb,ub])
#    E,sdos=slv_sdos.get_sdos(V, dIdV, par[0], par[1], par[2],E_sampling=E_sampling,rcond=rcond,return_matrix=False)
#    return E,sdos,par

