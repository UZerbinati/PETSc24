Using PETSc PC inside of NGSolve
=================================

In this tutorial we explore using `PETSc PC` as a preconditioner inside NGSolve preconditioning infrastructure.
Once again, we begin by creating a discretisation of the Poisson problem using H1 elements, in particular we consider the usual variational formulation
.. math::

   \text{find } u\in H^1_0(\Omega) \text{ s.t. } a(u,v) := \int_{\Omega} \nabla u\cdot \nabla v \; d\vec{x} = L(v) := \int_{\Omega} fv\; d\vec{x}\qquad v\in H^1_0(\Omega).

Such a discretisation can easily be constructed using NGSolve as follows: ::

   from ngsolve import *
   import netgen.gui
   import netgen.meshing as ngm
   from mpi4py.MPI import COMM_WORLD

   if COMM_WORLD.rank == 0:
      mesh = Mesh(unit_square.GenerateMesh(maxh=0.2).Distribute(COMM_WORLD))
   else:
      mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
   for _ in range(4):
      mesh.Refine()
   fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
   u,v = fes.TnT()
   a = BilinearForm(grad(u)*grad(v)*dx)
   f = LinearForm(fes)
   f += 32 * (y*(1-y)+x*(1-x)) * v * dx
   a.Assemble()
   f.Assemble()

We now consturct an NGSolve preconditioner wrapping a `PETSc PC`, in particular we will construct an Algebraic MultiGrid preconditioner using `HYPRE` and use the Krylov solver implemented inside NGSolve to solve the linear system. ::

   from ngsPETSc import pc
   from ngsolve.krylovspace import CG
   pre = Preconditioner(a, "PETScPC", pc_type="gamg")
   gfu = GridFunction(fes)
   gfu.vec.data = CG(a.mat, rhs=f.vec, pre=pre.mat, printrates=True)
   Draw(gfu)

We can use PETSc preconditioner as one of the building blocks of a more complex preconditioner. For example, we can use it a two-level additive Schwarz preconditioner.
In this case, we will use as fine space correction, the inverse of the local matrices associated with the patch of a vertex. ::

   def VertexPatchBlocks(mesh, fes):
    blocks = []
    freedofs = fes.FreeDofs()
    for v in mesh.vertices:
        vdofs = set()
        for el in mesh[v].elements:
            vdofs |= set(d for d in fes.GetDofNrs(el)
                         if freedofs[d])
        blocks.append(vdofs)
    return blocks

   blocks = VertexPatchBlocks(mesh, fes)
   blockjac = a.mat.CreateBlockSmoother(blocks)

   mgpre = pre.mat + blockjac
   gfu.vec.data = CG(a.mat, rhs=f.vec, pre=mgpre, printrates=True)
   Draw(gfu)



