from ngsolve import *
import netgen.gui
from netgen.geom2d import SplineGeometry
from ngsPETSc import NonLinearSolver
from mpi4py.MPI import COMM_WORLD

if COMM_WORLD.rank == 0:
    geo = SplineGeometry()
    pnums = [ geo.AddPoint (x,y,maxh=0.01) for x,y in [(0,0), (1,0), (1,0.1), (0,0.1)] ]
    for p1,p2,bc in [(0,1,"bot"), (1,2,"right"), (2,3,"top"), (3,0,"left")]:
        geo.Append(["line", pnums[p1], pnums[p2]], bc=bc)
    mesh = Mesh(geo.GenerateMesh(maxh=0.05).Distribute(COMM_WORLD))
else:
    mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
# E module and poisson number:
E, nu = 210, 0.2
# Lamé constants:
mu  = E / 2 / (1+nu)
lam = E * nu / ((1+nu)*(1-2*nu))
fes = VectorH1(mesh, order=2, dirichlet="left")
#gravity:
force = CoefficientFunction( (0,-1) )
u,_ = fes.TnT()
def Pow(a, b):
    return exp (log(a)*b)

def NeoHook (C):
    return 0.5 * mu * (Trace(C-I) + 2*mu/lam * Pow(Det(C), -lam/2/mu) - 1)

I = Id(mesh.dim)
F = I + Grad(u)
C = F.trans * F
factor = Parameter(0.1)
a = BilinearForm(fes, symmetric=True)
a += Variation(NeoHook (C).Compile() * dx
                -factor * (InnerProduct(force,u) ).Compile() * dx)
solver = NonLinearSolver(fes, a=a,
                            solverParameters={"snes_type": "newtonls",
                                            "snes_max_it": 10,
                                            "snes_monitor": "",
                                            "pc_type": "lu"})
gfu0 = GridFunction(fes)
gfu0.Set((0,0)) # initial guess
gfu = solver.solve(gfu0)
Draw(gfu, mesh, "displacement")