from netgen.csg import *
import netgen.gui
from ngsolve import *
from ngsolve.internal import visoptions
from mpi4py import MPI

order = 3

geo = CSGeometry()
cyl   = Cylinder(Pnt(0,0,0),Pnt(1,0,0),0.4).bc("cyl")
left  = Plane(Pnt(0,0,0), Vec(-1,0,0))
right = Plane(Pnt(1,0,0), Vec(1,0,0))
finitecyl = cyl * left * right
geo.AddSurface(cyl, finitecyl)
geo.NameEdge(cyl,left, "left")
geo.NameEdge(cyl,right, "right")

if MPI.COMM_WORLD.rank == 0:
    mesh = Mesh(geo.GenerateMesh(maxh=0.3).Distribute(MPI.COMM_WORLD))
else:
    mesh = Mesh(ngm.Mesh.Receive(MPI.COMM_WORLD))

mesh.Curve(order)

fes1 = VectorH1(mesh, order=order, dirichlet_bbnd="left")
fes = fes1*fes1
u,beta = fes.TrialFunction()

nsurf = specialcf.normal(3)

thickness = 0.1
Ptau = Id(3) - OuterProduct(nsurf,nsurf)
Ftau = grad(u).Trace() + Ptau
Ctautau = Ftau.trans * Ftau
Etautau = 0.5*(Ctautau - Ptau)

eps_beta = Sym(Ptau*grad(beta).Trace())
gradu = grad(u).Trace()
ngradu = gradu.trans*nsurf
#Average normal vector for affine geometry
if order == 1:
    gfn = GridFunction(fes1)
    gfn.Set(nsurf,definedon=mesh.Boundaries(".*"))
else:
    gfn = nsurf

a = BilinearForm(fes, symmetric=True)
#membrane energy
a += Variation( thickness*InnerProduct(Etautau, Etautau)*ds )
#bending energy
a += Variation( 0.5*thickness**3*InnerProduct(eps_beta-Sym(gradu.trans*grad(gfn)),eps_beta-Sym(gradu.trans*grad(gfn)))*ds )
#shearing energy
a += Variation( thickness*(ngradu-beta)*(ngradu-beta)*ds )

# external force
factor = Parameter(0.0)
a += Variation( -thickness*factor*y*u[1]*ds )

gfu = GridFunction(fes)

from ngsPETSc import NonLinearSolver
for loadstep in range(6):
    print("loadstep ", loadstep)
    factor.Set (1.5*(loadstep+1))
    solver = NonLinearSolver(fes, a=a,
                                solverParameters={"snes_type": "newtonls",
                                                "snes_max_it": 10,
                                                "snes_monitor": "",
                                                "ksp_monitor": "",
                                                "pc_type": "lu"})
    
    gfu = solver.solve(gfu)
Draw(gfu.components[0], mesh, "displacement")
Draw(gfu.components[1], mesh, "rotation")