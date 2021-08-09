var documenterSearchIndex = {"docs":
[{"location":"lie_group_solvers.html#Lie-group-solvers","page":"Lie group action solvers","title":"Lie group solvers","text":"","category":"section"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"An initial value problem manifold ordinary differential equation in the Lie action formulation.","category":"page"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"A Lie ODE on manifold M is defined in terms a vector field F (M  P  ℝ) to 𝔤 where 𝔤 is the Lie algebra of a Lie group G acting on M, with an initial value y_0 and P is the space of constant parameters. A solution to this problem is a curve ycolon ℝto M such that y(0)=y_0 and for each t in 0 T we have D_t y(t) = f(y(t) p t)circ y(t), where the circ is defined as","category":"page"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"Xcirc m = fracddtvert_t=0 exp(tZ)cdot m","category":"page"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"and cdot is the group action of G on M.","category":"page"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"The Lie group G must act transitively on M, that is for each pair of points p q on M there is an element a in G such that acdot p = q. See for example [CelledoniMarthinsenOwren2014] for details.","category":"page"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"[CelledoniMarthinsenOwren2014]: E. Celledoni, H. Marthinsen, and B. Owren, “An introduction to Lie group integrators – basics, new developments and applications,” Journal of Computational Physics, vol. 257, pp. 1040–1061, Jan. 2014, doi: 10.1016/j.jcp.2012.12.031.","category":"page"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"Modules = [ManifoldDiffEq]\nPages = [\"lie_solvers.jl\"]\nOrder = [:type, :function]","category":"page"},{"location":"lie_group_solvers.html#ManifoldDiffEq.ManifoldLieEuler","page":"Lie group action solvers","title":"ManifoldDiffEq.ManifoldLieEuler","text":"ManifoldLieEuler\n\nThe manifold Lie-Euler algorithm for problems in the LieODEProblemType formulation.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.ManifoldLieEulerCache","page":"Lie group action solvers","title":"ManifoldDiffEq.ManifoldLieEulerCache","text":"ManifoldLieEulerCache\n\nCache for ManifoldLieEuler.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.ManifoldLieEulerConstantCache","page":"Lie group action solvers","title":"ManifoldDiffEq.ManifoldLieEulerConstantCache","text":"ManifoldLieEulerConstantCache\n\nConstant cache for ManifoldLieEuler.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.RKMK4","page":"Lie group action solvers","title":"ManifoldDiffEq.RKMK4","text":"RKMK4\n\nThe Lie group variant of fourth-order Runge-Kutta algorithm for problems in the LieODEProblemType formulation. The tableau is:\n\n0    | 0\n1/2  | 1/2  0\n1/2  | 0    1/2  0\n1    | 0    0    1    0\n------------------------------\n     | 1/6  1/3  1/3  1/6\n\nFor more details see [MuntheKaasOwren1999].\n\n[MuntheKaasOwren1999]: H. Munthe–Kaas and B. Owren, “Computations in a free Lie algebra,” Philosophical Transactions of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences, vol. 357, no. 1754, pp. 957–981, Apr. 1999, doi: 10.1098/rsta.1999.0361.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.RKMK4Cache","page":"Lie group action solvers","title":"ManifoldDiffEq.RKMK4Cache","text":"RKMK4Cache\n\nCache for RKMK4.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.RKMK4ConstantCache","page":"Lie group action solvers","title":"ManifoldDiffEq.RKMK4ConstantCache","text":"RKMK4ConstantCache\n\nConstant cache for RKMK4.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.apply_diff_group-Tuple{AbstractGroupAction, Any, Any, Any}","page":"Lie group action solvers","title":"ManifoldDiffEq.apply_diff_group","text":"apply_diff_group(A::AbstractGroupAction, a, X, p)\n\nFor a point on manifold p  mathcal M and an element X of the tangent space at a, an element of the Lie group of action A, X  T_a mathcal G, compute the differential of action of a on p for vector X, as specified by rule A. When action on element p is written as mathrmdτ^p, with the specified left or right convention, the differential transforms vectors\n\n(mathrmdτ^p)  T_a mathcal G  T_τ_a p mathcal M\n\n\n\n\n\n","category":"method"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"ManifoldDiffEq.LieODEProblemType\nManifoldDiffEq.LieManifoldDiffEqOperator","category":"page"},{"location":"lie_group_solvers.html#ManifoldDiffEq.LieODEProblemType","page":"Lie group action solvers","title":"ManifoldDiffEq.LieODEProblemType","text":"LieODEProblemType\n\nAn initial value problem manifold ordinary differential equation in the Lie action formulation.\n\nA Lie ODE on manifold M is defined in terms a vector field F (ℝ  P  M) to 𝔤 where 𝔤 is the Lie algebra of a Lie group G acting on M, with an initial value y₀ and P is the space of constant parameters. A solution to this problem is a curve yℝto M such that y(0)=y₀ and for each t  0 T we have D_t y(t) = f(t y(t))y(t), where the  is defined as\n\nXm = fracddtvert_t=0 exp(tZ)m\n\nand  is the group action of G on M.\n\nnote: Note\nProofs of convergence and order have several assumptions, including time-independence of F. Integrators may not work well if these assumptions do not hold.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.LieManifoldDiffEqOperator","page":"Lie group action solvers","title":"ManifoldDiffEq.LieManifoldDiffEqOperator","text":"LieManifoldDiffEqOperator{T<:Number,TF} <: AbstractDiffEqOperator{T}\n\nDiffEq operator on manifolds in the Lie group action formulation.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#Frozen-coefficients-solvers","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"","category":"section"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"An initial value problem manifold ordinary differential equation in the frozen coefficients formulation by Crouch and Grossman, see [CrouchGrossman1993].","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"A frozen coefficients ODE on manifold M is defined in terms a vector field Fcolon (M  P  ℝ) to T_p M where p is the point given as the third argument to F, with an initial value y_0 and P is the space of constant parameters. Frozen coefficients mean that we also have means to transport a vector X in T_p M obtained from F to a different point on a manifold or a different time (parameters are assumed to be constant). This is performed through operator_vector_transport, an object of a subtype of AbstractVectorTransportOperator, stored in FrozenManifoldDiffEqOperator.","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"A solution to this problem is a curve ycolon ℝto M such that y(0)=y_0 and for each t in 0 T we have D_t y(t) = F(y(t) p t).","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"The problem is usually studied for manifolds that are Lie groups or homogeneous manifolds, see[CelledoniMarthinsenOwren2014].","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"[CrouchGrossman1993]: P. E. Crouch and R. Grossman, “Numerical integration of ordinary differential equations on manifolds,” J Nonlinear Sci, vol. 3, no. 1, pp. 1–33, Dec. 1993, doi: 10.1007/BF02429858.","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"[CelledoniMarthinsenOwren2014]: E. Celledoni, H. Marthinsen, and B. Owren, “An introduction to Lie group integrators – basics, new developments and applications,” Journal of Computational Physics, vol. 257, pp. 1040–1061, Jan. 2014, doi: 10.1016/j.jcp.2012.12.031.","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"Modules = [ManifoldDiffEq]\nPages = [\"frozen_solvers.jl\"]\nOrder = [:type, :function]","category":"page"},{"location":"frozen_coefficients.html#ManifoldDiffEq.CG2","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.CG2","text":"CG2\n\nA Crouch-Grossmann algorithm of second order for problems in the ExplicitManifoldODEProblemType formulation. See order 2 conditions discussed in [OwrenMarthinsen1999]. Tableau:\n\n0    | 0\n1/2  | 1/2  0\n----------------\n     | 0    1\n\n[OwrenMarthinsen1999]: B. Owren and A. Marthinsen, “Runge-Kutta Methods Adapted to Manifolds and Based on Rigid Frames,” BIT Numerical Mathematics, vol. 39, no. 1, pp. 116–142, Mar. 1999, doi: 10.1023/A:1022325426017.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.CG2Cache","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.CG2Cache","text":"CG2Cache\n\nCache for CG2.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.ManifoldEuler","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.ManifoldEuler","text":"ManifoldEuler\n\nThe manifold Euler algorithm for problems in the ExplicitManifoldODEProblemType formulation.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.ManifoldEulerCache","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.ManifoldEulerCache","text":"ManifoldEulerCache\n\nCache for ManifoldEuler.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.ManifoldEulerConstantCache","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.ManifoldEulerConstantCache","text":"ManifoldEulerConstantCache\n\nCache for ManifoldEuler.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"ManifoldDiffEq.ExplicitManifoldODEProblemType\nManifoldDiffEq.FrozenManifoldDiffEqOperator\nManifoldDiffEq.AbstractVectorTransportOperator","category":"page"},{"location":"frozen_coefficients.html#ManifoldDiffEq.ExplicitManifoldODEProblemType","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.ExplicitManifoldODEProblemType","text":"ExplicitManifoldODEProblemType\n\nAn initial value problem manifold ordinary differential equation in the frozen coefficients formulation by Crouch and Grossman, see [CrouchGrossman1993].\n\nA frozen coefficients ODE on manifold M is defined in terms a vector field F (M  P  ℝ) to T_p M where p is the point given as the third argument to F, with an initial value y₀ and P is the space of constant parameters. A solution to this problem is a curve yℝto M such that y(0)=y₀ and for each t  0 T we have D_t y(t) = F(t p y(t)),\n\nnote: Note\nProofs of convergence and order have several assumptions, including time-independence of F. Integrators may not work well if these assumptions do not hold.\n\n[CrouchGrossman1993]: P. E. Crouch and R. Grossman, “Numerical integration of ordinary differential equations on manifolds,” J Nonlinear Sci, vol. 3, no. 1, pp. 1–33, Dec. 1993, doi: 10.1007/BF02429858.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.FrozenManifoldDiffEqOperator","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.FrozenManifoldDiffEqOperator","text":"FrozenManifoldDiffEqOperator{T<:Number,TM<:AbstractManifold,TF,TVT} <: SciMLBase.AbstractDiffEqOperator{T}\n\nDiffEq operator on manifolds in the frozen vector field formulation\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.AbstractVectorTransportOperator","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.AbstractVectorTransportOperator","text":"AbstractVectorTransportOperator\n\nAbstract type for vector transport operators in the frozen coefficients formulation.\n\n\n\n\n\n","category":"type"},{"location":"index.html#ManifoldDiffEq","page":"Home","title":"ManifoldDiffEq","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"The package ManifoldDiffEq aims to provide a library of differential equation solvers on manifolds. The library is built on top of Manifolds.jl and follows the interface of OrdinaryDiffEq.jl.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"The code below demonstrates usage of ManifoldDiffEq to solve a simple equation and visualize the results.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Methods implemented in this library are described in, for example:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"E. Hairer, C. Lubich, and G. Wanner, Geometric Numerical Integration: Structure-Preserving\nAlgorithms for Ordinary Differential Equations, 2nd ed. 2006. 2nd printing 2010 edition. Heidelberg ;\nNew York: Springer, 2010.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"using GLMakie, LinearAlgebra\n\nn = 10\n\nθ = [0;(0.5:n-0.5)/n;1]\nφ = [(0:2n-2)*2/(2n-1);2]\nx = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]\ny = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]\nz = [cospi(θ) for θ in θ, φ in φ]\n\nfunction f2(x, y, z)\n    p = [x, y, z]\n    return exp(x) * cross(p, [1.0, 0.0, 0.0]) + exp(y) * cross(p, [0.0, 1.0, 0.0])\nend\n\ntans = f2.(vec(x), vec(y), vec(z))\nu = [a[1] for a in tans]\nv = [a[2] for a in tans]\nw = [a[3] for a in tans]\n\nscene = Scene();\n\narr = GLMakie.arrows(\n    vec(x), vec(y), vec(z), u, v, w;\n    arrowsize = 0.1, linecolor = (:gray, 0.7), linewidth = 0.02, lengthscale = 0.1\n)\n\nusing ManifoldDiffEq, OrdinaryDiffEq, Manifolds\n\n# This is the same ODE problem on two different formulations: Lie group action (prob_lie)\n# and frozen coefficients (prob_frozen)\nS2 = Manifolds.Sphere(2)\n\nA_lie = ManifoldDiffEq.LieManifoldDiffEqOperator{Float64}() do u, p, t\n    return hat(SpecialOrthogonal(3), Matrix(I(3)), cross(u, f2(u...)))\nend\nprob_lie = ODEProblem(A_lie, [0.0, 1.0, 0.0], (0, 20.0))\n\nA_frozen = ManifoldDiffEq.FrozenManifoldDiffEqOperator{Float64}() do u, p, t\n    return f2(u...)\nend\nprob_frozen = ODEProblem(A_frozen, [0.0, 1.0, 0.0], (0, 20.0))\n\naction = RotationAction(Euclidean(3), SpecialOrthogonal(3))\nalg_lie_euler = ManifoldDiffEq.ManifoldLieEuler(S2, ExponentialRetraction(), action)\nalg_lie_rkmk4 = ManifoldDiffEq.RKMK4(S2, ExponentialRetraction(), action)\n\nalg_manifold_euler = ManifoldDiffEq.ManifoldEuler(S2, ExponentialRetraction())\nalg_cg2 = ManifoldDiffEq.CG2(S2, ExponentialRetraction())\nalg_cg3 = ManifoldDiffEq.CG3(S2, ExponentialRetraction())\n\ndt = 0.05\nsol_lie = solve(prob_lie, alg_lie_euler, dt = dt)\nsol_frozen_cg2 = solve(prob_frozen, alg_cg2, dt = dt)\nsol_frozen_cg3 = solve(prob_frozen, alg_cg3, dt = dt)\nsol_rkmk4 = solve(prob_lie, alg_lie_rkmk4, dt = dt)\n\nfor (sol, color) in [(sol_lie, :red), (sol_frozen_cg2, :green), (sol_frozen_cg3, :blue)]\n    GLMakie.lines!([u[1] for u in sol.u], [u[2] for u in sol.u], [u[3] for u in sol.u]; linewidth = 10, color=color)\nend","category":"page"}]
}
