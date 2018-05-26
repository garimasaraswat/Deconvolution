import sympy as sp
import symfn as symfn

fn_fermi=sp.lambdify((symfn.E,symfn.V, symfn.T),symfn.fermi(),"numpy")
fn_dfermi=sp.lambdify((symfn.E,symfn.V, symfn.T),(symfn.fermi()).diff(symfn.V,1),"numpy")

fn_bcs_dos=sp.lambdify((symfn.E,symfn.V, symfn.gamma, symfn.delta),symfn.bcs_dos(),"numpy")
fn_dbcs_dos=sp.lambdify((symfn.E,symfn.V, symfn.gamma, symfn.delta),symfn.dbcs_dos(),"numpy")