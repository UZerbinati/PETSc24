from ngsolve import *
from netgen.geom2d import SplineGeometry
import netgen.gui

t = 0.1
E, nu, k = 10.92, 0.3, 5/6
G = E/(2*(1+nu))
R = 5
fz = 1
sg = SplineGeometry()
pnts = [ (0,0), (R,0), (R,R), (0,R) ]
pind = [ sg.AppendPoint(*pnt) for pnt in pnts ]
sg.Append(['line',pind[0],pind[1]], leftdomain=1, rightdomain=0, bc="bottom")
sg.Append(['spline3',pind[1],pind[2],pind[3]], leftdomain=1, rightdomain=0, bc="circ")
sg.Append(['line',pind[3],pind[0]], leftdomain=1, rightdomain=0, bc="left")
mesh = Mesh(sg.GenerateMesh(maxh=R/3))
mesh.Curve(3)
Draw(mesh)
n = specialcf.normal(2)
def CMatInv(mat, E, nu):
    return (1+nu)/E*(mat-nu/(nu+1)*Trace(mat)*Id(2))
order=1
fesB = HCurl(mesh, order=order-1, dirichlet="circ", autoupdate=True)
fesS = HDivDiv(mesh, order=order-1, dirichlet="", autoupdate=True)
fesW = H1(mesh, order=order, dirichlet="circ", autoupdate=True)

fes = FESpace( [fesW, fesB, fesS], autoupdate=True )
(u,beta,sigma), (du,dbeta,dsigma) = fes.TnT()
a = BilinearForm(fes)
a += (-12/t**3*InnerProduct(CMatInv(sigma, E, nu),dsigma) + InnerProduct(dsigma,grad(beta)) + InnerProduct(sigma,grad(dbeta)))*dx
a += ( -((sigma*n)*n)*(dbeta*n) - ((dsigma*n)*n)*(beta*n) )*dx(element_boundary=True)
a += t*k*G*InnerProduct( grad(u)-beta, grad(du)-dbeta )*dx

f = LinearForm(fes)
f += -fz*du*dx

gfsol = GridFunction(fes, autoupdate=True)
gfu, gfbeta, gfsigma = gfsol.components

def SolveBVP():
    fes.Update()
    gfsol.Update()
    with TaskManager():
        a.Assemble()
        f.Assemble()
        inv = a.mat.Inverse(fes.FreeDofs())
        gfsol.vec.data = inv * f.vec
l = []
for i in range(5):
    print("i = ", i)
    SolveBVP()
    mesh.Refine()

Draw(gfsol.components[0], mesh, "displacement")