import sympy as sp
import constants as cnst

E,V,T,gamma,delta= sp.symbols("E V T gamma delta",real=True)

def fermi():
   return 1/(sp.exp((E-V)/(cnst.k_BeV*T))+1)


def bcs_dos():
    return sp.sign(E-V)*sp.re((E-V)/sp.sqrt((E-V)**2.+2.*sp.I*gamma*(E-V) -delta**2.))


def dbcs_dos():
    y=(E-V)/sp.sqrt((E-V)**2.+2.*sp.I*gamma*(E-V) -delta**2.)
    return sp.re(y.diff(V,1)*sp.sign(E-V))

# Added comment : These are Aditya's codes
