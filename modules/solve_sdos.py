import numpy as np
import numfn as numfn

def get_mat(V,gamma,delta,T,E_sampling=3000):
    E=np.linspace(min(V),max(V),E_sampling)
    C=np.zeros((np.size(V),E_sampling),np.float64)
    for i,vi in enumerate(V):
        C[i,:]=numfn.fn_dbcs_dos(E,vi,gamma,delta)*(numfn.fn_fermi(E,vi,T)-numfn.fn_fermi(E,0.,T))+numfn.fn_dfermi(E,vi,T)*numfn.fn_bcs_dos(E,vi,gamma,delta)
    return E,C*np.mean(np.diff(E))

def get_sdos(V, dIdV, gamma, delta, T, E_sampling=3000,rcond=1e-5, return_matrix=False):
    E,C=get_mat(V,gamma,delta,T,E_sampling=E_sampling)
    M=np.transpose(np.matrix(C))*np.matrix(C) ; M_inv=np.linalg.pinv(M,rcond=rcond)
    Y=np.transpose(np.matrix(C))*np.transpose(np.matrix(dIdV))
    sdos=np.array(M_inv*Y).flatten()
    if return_matrix:
        return sdos,C
    else:
        return E,sdos,C 


