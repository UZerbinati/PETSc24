from ngsolve import *
from netgen.geom2d import unit_square
import petsc4py.PETSc as psc
from mpi4py import MPI
from time import time
from ngsPETSc import pc, NullSpace, KrylovSolver

import netgen.gui

ts = time()
comm = MPI.COMM_WORLD
import netgen.meshing as ngm

if comm.rank == 0:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(comm))
else:
    mesh = Mesh(ngm.Mesh.Receive(comm))

E, nu = 210, 0.2
mu  = E / 2 / (1+nu)
lam = E * nu / ((1+nu)*(1-2*nu))

def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(2)

fes = VectorH1(mesh, order=1)
u,v = fes.TnT()

# Rigid body motion
values = [CF((1,0)), CF((0,1)), CF((-y,x))]
rbms = ["constant"]
for val in values:
    rbm = GridFunction(fes)
    rbm.Set(val)
    rbms.append(rbm.vec)
nullspace = NullSpace(fes, rbms, near=True)

with TaskManager():
    a = BilinearForm(InnerProduct(Stress(Sym(Grad(u))), Sym(Grad(v)))*dx)
    a.Assemble()

force = CF( (0,1e-3) )
f = LinearForm(force*v*ds("right")).Assemble()


solver = KrylovSolver(a,fes, solverParameters={'ksp_type': 'cg', 'pc_type': 'gamg'}, nullspace=nullspace)
gfu = solver.solve(f)

Draw (gfu)