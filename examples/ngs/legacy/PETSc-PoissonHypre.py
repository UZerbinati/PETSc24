# mpirun -np 4 python3 PETSc-Poisson.py

from ngsolve import *
from netgen.geom2d import unit_square
import petsc4py.PETSc as psc
from mpi4py import MPI
from time import time

ts = time()
comm = MPI.COMM_WORLD
import netgen.meshing as ngm

if comm.rank == 0:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1).Distribute(comm))
else:
    mesh = Mesh(ngm.Mesh.Receive(comm))

fes = H1(mesh, order=1, dirichlet=".*")
printonce ("ndof = {}".format(fes.ndofglobal))
u,v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(1*v*dx).Assemble()

comm.Barrier()
printonce ("have matrix, t = {} ".format(time()-ts))

from ngsPETSc import pc
pre = Preconditioner(a, "PETScPC", pc_type="hypre", pc_hypre_boomeramg_agg_nl="1")

print ("type pre = {}".format(type(pre)))

comm.Barrier()
printonce ("have pre   , t = {}".format(time()-ts))


gfu = GridFunction(fes)


from ngsolve.krylovspace import CGSolver
inv = CGSolver(a.mat, pre, printrates=comm.rank==0)
gfu.vec.data = inv * f.vec

comm.Barrier()
printonce ("have sol   , t ={}".format(time()-ts)) 


printonce("Done")
