var documenterSearchIndex = {"docs":
[{"location":"references.html#Literature","page":"References","title":"Literature","text":"","category":"section"},{"location":"references.html","page":"References","title":"References","text":"E. Celledoni, H. Marthinsen and B. Owren. An introduction to Lie group integrators – basics, new developments and applications. Journal of Computational Physics 257, 1040–1061 (2014). Accessed on Jul 14, 2021, arXiv: 1207.0069.\n\n\n\nP. E. Crouch and R. Grossman. Numerical integration of ordinary differential equations on manifolds. Journal of Nonlinear Science 3, 1–33 (1993). Accessed on Jul 17, 2021.\n\n\n\nK. Engø and A. Marthinsen. Modeling and Solution of Some Mechanical Problems on Lie Groups. Multibody System Dynamics 2, 71–88 (1998). Accessed on Oct 13, 2021.\n\n\n\nE. Hairer, C. Lubich and G. Wanner. Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations. 2nd ed. 2006. 2nd printing 2010 edition Edition (Springer, Heidelberg ; New York, 2010).\n\n\n\nZ. Jackiewicz, A. Marthinsen and B. Owren. Construction of Runge–Kutta methods of Crouch–Grossman type of high order. Advances in Computational Mathematics 13, 405–415 (2000). Accessed on Oct 15, 2021.\n\n\n\nH. Munthe–Kaas and B. Owren. Computations in a free Lie algebra. Philosophical Transactions of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences 357, 957–981 (1999). Accessed on Jul 14, 2021. Publisher: Royal Society.\n\n\n\nB. Owren and A. Marthinsen. Runge-Kutta Methods Adapted to Manifolds and Based on Rigid Frames. BIT Numerical Mathematics 39, 116–142 (1999). Accessed on Jul 15, 2021.\n\n\n\n","category":"page"},{"location":"lie_group_solvers.html#Lie-group-solvers","page":"Lie group action solvers","title":"Lie group solvers","text":"","category":"section"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"An initial value problem manifold ordinary differential equation in the Lie action formulation.","category":"page"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"A Lie ODE on manifold M is defined in terms a vector field F (M  P  ℝ) to 𝔤 where 𝔤 is the Lie algebra of a Lie group G acting on M, with an initial value y_0 and P is the space of constant parameters. A solution to this problem is a curve ycolon ℝto M such that y(0)=y_0 and for each t in 0 T we have D_t y(t) = f(y(t) p t)circ y(t), where the circ is defined as","category":"page"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"Xcirc m = fracddtvert_t=0 exp(tZ)cdot m","category":"page"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"and cdot is the group action of G on M.","category":"page"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"The Lie group G must act transitively on M, that is for each pair of points p q on M there is an element a in G such that acdot p = q. See for example [CMO14] for details.","category":"page"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"Modules = [ManifoldDiffEq]\nPages = [\"lie_solvers.jl\"]\nOrder = [:type, :function]","category":"page"},{"location":"lie_group_solvers.html#ManifoldDiffEq.ManifoldLieEuler","page":"Lie group action solvers","title":"ManifoldDiffEq.ManifoldLieEuler","text":"ManifoldLieEuler\n\nThe manifold Lie-Euler algorithm for problems in the LieODEProblemType formulation.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.ManifoldLieEulerCache","page":"Lie group action solvers","title":"ManifoldDiffEq.ManifoldLieEulerCache","text":"ManifoldLieEulerCache\n\nCache for ManifoldLieEuler.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.ManifoldLieEulerConstantCache","page":"Lie group action solvers","title":"ManifoldDiffEq.ManifoldLieEulerConstantCache","text":"ManifoldLieEulerConstantCache\n\nConstant cache for ManifoldLieEuler.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.RKMK4","page":"Lie group action solvers","title":"ManifoldDiffEq.RKMK4","text":"RKMK4\n\nThe Lie group variant of fourth-order Runge-Kutta algorithm for problems in the LieODEProblemType formulation, called Runge-Kutta Munthe-Kaas. The Butcher tableau is:\n\nbeginarrayccccc\n0  0 \nfrac12  0  frac12  0 \nfrac12  frac12  0 \n1  0  0  1  0\nhline\n frac16  frac13  frac16  frac16\nendarray\n\nFor more details see [MO99].\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.RKMK4Cache","page":"Lie group action solvers","title":"ManifoldDiffEq.RKMK4Cache","text":"RKMK4Cache\n\nCache for RKMK4.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.RKMK4ConstantCache","page":"Lie group action solvers","title":"ManifoldDiffEq.RKMK4ConstantCache","text":"RKMK4ConstantCache\n\nConstant cache for RKMK4.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html","page":"Lie group action solvers","title":"Lie group action solvers","text":"ManifoldDiffEq.LieODEProblemType\nManifoldDiffEq.LieManifoldDiffEqOperator","category":"page"},{"location":"lie_group_solvers.html#ManifoldDiffEq.LieODEProblemType","page":"Lie group action solvers","title":"ManifoldDiffEq.LieODEProblemType","text":"LieODEProblemType\n\nAn initial value problem manifold ordinary differential equation in the Lie action formulation.\n\nA Lie ODE on manifold M is defined in terms a vector field F (ℝ  P  M) to 𝔤 where 𝔤 is the Lie algebra of a Lie group G acting on M, with an initial value y₀ and P is the space of constant parameters. A solution to this problem is a curve yℝto M such that y(0)=y₀ and for each t  0 T we have D_t y(t) = F(y(t) p t)y(t), where the  is defined as\n\nXm = fracddtvert_t=0 exp(tZ)m\n\nand  is the group action of G on M.\n\nnote: Note\nProofs of convergence and order have several assumptions, including time-independence of F. Integrators may not work well if these assumptions do not hold.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#ManifoldDiffEq.LieManifoldDiffEqOperator","page":"Lie group action solvers","title":"ManifoldDiffEq.LieManifoldDiffEqOperator","text":"LieManifoldDiffEqOperator{T<:Number,TF} <: AbstractDiffEqOperator{T}\n\nDiffEq operator on manifolds in the Lie group action formulation.\n\n\n\n\n\n","category":"type"},{"location":"lie_group_solvers.html#Literature","page":"Lie group action solvers","title":"Literature","text":"","category":"section"},{"location":"frozen_coefficients.html#Frozen-coefficients-solvers","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"","category":"section"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"An initial value problem manifold ordinary differential equation in the frozen coefficients formulation by Crouch and Grossman, see [CG93].","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"A frozen coefficients ODE on manifold M is defined in terms a vector field Fcolon (M  P  ℝ) to T_p M where p is the point given as the third argument to F, with an initial value y_0 and P is the space of constant parameters. Frozen coefficients mean that we also have means to transport a vector X in T_p M obtained from F to a different point on a manifold or a different time (parameters are assumed to be constant). This is performed through operator_vector_transport, an object of a subtype of AbstractVectorTransportOperator, stored in FrozenManifoldDiffEqOperator.","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"A solution to this problem is a curve ycolon ℝto M such that y(0)=y_0 and for each t in 0 T we have D_t y(t) = F(y(t) p t).","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"The problem is usually studied for manifolds that are Lie groups or homogeneous manifolds, see[CMO14].","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"Note that in this formulation s-stage explicit Runge-Kutta schemes that for mathbbR^n are defined by equations","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"beginalign*\nX_1 = f(u_n p t) \nX_2 = f(u_n+h a_21 X_1 p t+c_2 h) \nX_3 = f(u_n+h a_31 X_1 + a_32 X_2 p t+c_3 h) \nvdots \nX_s = f(u_n+h a_s1 X_1 + a_s2 X_2 + dots + a_ss-1 X_s-1 p t+c_s h) \nu_n+1 = u_n + hsum_i=1^s b_i X_i\nendalign*","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"for general manifolds read","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"beginalign*\nX_1 = f(u_n p t) \nu_n21 = exp_u_n(h a_21 X_1) \nX_2 = f(u_n21 p t+c_2 h) \nu_n31 = exp_u_n(h a_31 X_1) \nu_n32 = exp_u_n31(mathcal P_u_n31gets u_n21 h a_32 X_2) \nX_3 = f(u_n32 p t+c_3 h) \nvdots \nX_s = f(u_nss-1 p t+c_s h) \nX_b1 = X_1 \nu_b1 = exp_u_n(h b_1 X_b1) \nX_b2 = mathcal P_u_b1 gets u_n21 X_2 \nu_b2 = exp_u_b1(h b_2 X_b2) \nvdots \nX_bs = mathcal P_u_bs-1 gets u_nss-1 X_s \nu_n+1 = exp_u_bs-1(h b_s X_bs)\nendalign*","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"Vector transports correspond to handling frozen coefficients. Note that the implementation allows for easy substitution of methods used for calculation of the exponential map (for example to use an approximation) and vector transport (if the default vector transport is not suitable for the problem). It is desirable to use a flat vector transport instead of a torsion-free one when available, for example the plus or minus Cartan-Schouten connections on Lie groups.","category":"page"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"Modules = [ManifoldDiffEq]\nPages = [\"frozen_solvers.jl\"]\nOrder = [:type, :function]","category":"page"},{"location":"frozen_coefficients.html#ManifoldDiffEq.CG2","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.CG2","text":"CG2\n\nA Crouch-Grossmann algorithm of second order for problems in the ExplicitManifoldODEProblemType formulation. The Butcher tableau is identical to the Euclidean RK2:\n\nbeginarrayccc\n0  0 \nfrac12  frac12  0 \nhline\n 0  1\nendarray\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.CG2Cache","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.CG2Cache","text":"CG2Cache\n\nCache for CG2.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.CG2_3","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.CG2_3","text":"CG2_3\n\nA Crouch-Grossmann algorithm of order 2(3) for problems in the ExplicitManifoldODEProblemType formulation. The Butcher tableau reads (see tableau (5) of [EM98]):\n\nbeginarraycccc\n0  0 \nfrac34  frac34  0 \nfrac1724  frac119216  frac17108  0\nhline\n frac34  frac314  frac-152\n frac1351  -frac23  frac2417\nendarray\n\nThe last row is used for error estimation.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.CG2_3Cache","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.CG2_3Cache","text":"CG2_3Cache\n\nCache for CG2_3.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.CG3","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.CG3","text":"CG3\n\nA Crouch-Grossmann algorithm of second order for problems in the ExplicitManifoldODEProblemType formulation. See tableau 6.1 of [OM99]:\n\nbeginarraycccc\n0  0 \nfrac34  frac34  0 \nfrac1724  frac119216  frac17108  0\nhline\n frac1351  -frac23  frac2417\nendarray\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.CG3Cache","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.CG3Cache","text":"CG3Cache\n\nCache for CG3.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.CG4a","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.CG4a","text":"CG4a\n\nA Crouch-Grossmann algorithm of second order for problems in the ExplicitManifoldODEProblemType formulation. See coefficients from Example 1 of [JMO00].\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.CG4aCache","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.CG4aCache","text":"CG4aCache\n\nCache for CG4a.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.ManifoldEuler","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.ManifoldEuler","text":"ManifoldEuler\n\nThe manifold Euler algorithm for problems in the ExplicitManifoldODEProblemType formulation.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.ManifoldEulerCache","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.ManifoldEulerCache","text":"ManifoldEulerCache\n\nCache for ManifoldEuler.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.ManifoldEulerConstantCache","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.ManifoldEulerConstantCache","text":"ManifoldEulerConstantCache\n\nCache for ManifoldEuler.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html","page":"Frozen coefficients solvers","title":"Frozen coefficients solvers","text":"ManifoldDiffEq.ExplicitManifoldODEProblemType\nManifoldDiffEq.FrozenManifoldDiffEqOperator\nManifoldDiffEq.AbstractVectorTransportOperator\nManifoldDiffEq.DefaultVectorTransportOperator","category":"page"},{"location":"frozen_coefficients.html#ManifoldDiffEq.ExplicitManifoldODEProblemType","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.ExplicitManifoldODEProblemType","text":"ExplicitManifoldODEProblemType\n\nAn initial value problem manifold ordinary differential equation in the frozen coefficients formulation by Crouch and Grossman, see [CG93].\n\nA frozen coefficients ODE on manifold M is defined in terms a vector field F (M  P  ℝ) to T_p M where p is the point given as the third argument to F, with an initial value y₀ and P is the space of constant parameters. A solution to this problem is a curve yℝto M such that y(0)=y₀ and for each t  0 T we have D_t y(t) = F(y(t) p t),\n\nnote: Note\nProofs of convergence and order have several assumptions, including time-independence of F. Integrators may not work well if these assumptions do not hold.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.FrozenManifoldDiffEqOperator","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.FrozenManifoldDiffEqOperator","text":"FrozenManifoldDiffEqOperator{T<:Number,TM<:AbstractManifold,TF,TVT} <: SciMLBase.AbstractDiffEqOperator{T}\n\nDiffEq operator on manifolds in the frozen vector field formulation.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.AbstractVectorTransportOperator","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.AbstractVectorTransportOperator","text":"AbstractVectorTransportOperator\n\nAbstract type for vector transport operators in the frozen coefficients formulation.\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#ManifoldDiffEq.DefaultVectorTransportOperator","page":"Frozen coefficients solvers","title":"ManifoldDiffEq.DefaultVectorTransportOperator","text":"(vto::DefaultVectorTransportOperator)(M::AbstractManifold, p, X, q, params, t_from, t_to)\n\nIn the frozen coefficient formulation, transport tangent vector X such that X = f(p, params, t_from) to point q at time t_to. This provides a sort of estimation of f(q, params, t_to).\n\n\n\n\n\n","category":"type"},{"location":"frozen_coefficients.html#Literature","page":"Frozen coefficients solvers","title":"Literature","text":"","category":"section"},{"location":"examples.html#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples.html","page":"Examples","title":"Examples","text":"We take a look at the simple example from","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"In the following code an ODE on a sphere is solved the introductionary example from the lecture notes by E. Hairer.","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"We solve the ODE system on the sphere mathbb S^2 given by","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"beginpmatrix\n    dot x \n    dot y \n    dot z\nendpmatrix\n=\nbeginpmatrix\n    0  zI_3  -yI_2 \n    -zI_3  0  xI_1 \n    yI_2 -xI_1  0\nendpmatrix\nbeginpmatrix\n    x \n    y \n    z\nendpmatrix","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"using ManifoldDiffEq, OrdinaryDiffEq, Manifolds\nusing GLMakie, LinearAlgebra, Colors\n\nn = 25\n\nθ = [0;(0.5:n-0.5)/n;1]\nφ = [(0:2n-2)*2/(2n-1);2]\nx = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]\ny = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]\nz = [cospi(θ) for θ in θ, φ in φ]\n\nfunction f2(x, y, z)\n    Iv = [1.6, 1.0, 2/3]\n    p = [x, y, z]\n    A = [0 -z y; z 0 -x; -y x 0]\n    return A * (p./Iv)\nend\n\ntans = f2.(vec(x), vec(y), vec(z))\nu = [a[1] for a in tans]\nv = [a[2] for a in tans]\nw = [a[3] for a in tans]\n\nf = Figure();\nAxis3(f[1,1])\n\narr = GLMakie.arrows!(\n           vec(x), vec(y), vec(z), u, v, w;\n           arrowsize = 0.02, linecolor = (:gray, 0.7), linewidth = 0.0075, lengthscale = 0.1\n)\nsave(\"docs/src/assets/img/first_example_vector_field.png\", f)","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"which looks like","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"(Image: The ODE illustrated as a tangent vector field)","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"Let's set up the manifold, the sphere and two different types of problems/solvers A first one that uses the Lie group action of the Special orthogonal group acting on data with 2 solvers and direct solvers on the sphere, using 3 other solvers using the idea of frozen coefficients.","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"S2 = Manifolds.Sphere(2)\nu0 = [0.0, sqrt(9/10), sqrt(1/10)]\ntspan = (0, 20.0)\n\nA_lie = ManifoldDiffEq.LieManifoldDiffEqOperator{Float64}() do u, p, t\n    return hat(SpecialOrthogonal(3), Matrix(I(3)), cross(u, f2(u...)))\nend\nprob_lie = ManifoldDiffEq.ManifoldODEProblem(A_lie, u0, tspan, S2)\n\nA_frozen = ManifoldDiffEq.FrozenManifoldDiffEqOperator{Float64}() do u, p, t\n    return f2(u...)\nend\nprob_frozen = ManifoldDiffEq.ManifoldODEProblem(A_frozen, u0, tspan, S2)\n\naction = RotationAction(Euclidean(3), SpecialOrthogonal(3))\nalg_lie_euler = ManifoldDiffEq.ManifoldLieEuler(S2, ExponentialRetraction(), action)\nalg_lie_rkmk4 = ManifoldDiffEq.RKMK4(S2, ExponentialRetraction(), action)\n\nalg_manifold_euler = ManifoldDiffEq.ManifoldEuler(S2, ExponentialRetraction())\nalg_cg2 = ManifoldDiffEq.CG2(S2, ExponentialRetraction())\nalg_cg23 = ManifoldDiffEq.CG2_3(S2, ExponentialRetraction())\nalg_cg3 = ManifoldDiffEq.CG3(S2, ExponentialRetraction())\n\ndt = 0.1\nsol_lie = solve(prob_lie, alg_lie_euler, dt = dt)\nsol_rkmk4 = solve(prob_lie, alg_lie_rkmk4, dt = dt)\n\nsol_frozen = solve(prob_frozen, alg_manifold_euler, dt=dt)\nsol_frozen_cg2 = solve(prob_frozen, alg_cg2, dt = dt)\nsol_frozen_cg23 = solve(prob_frozen, alg_cg23)\nsol_frozen_cg3 = solve(prob_frozen, alg_cg3, dt = dt)\n\nplot_sol(sol, col) = GLMakie.lines!([u[1] for u in sol.u], [u[2] for u in sol.u], [u[3] for u in sol.u]; linewidth = 2, color=col)\n\nl1 = plot_sol(sol_lie, colorant\"#999933\")\nl2 = plot_sol(sol_rkmk4, colorant\"#DDCC77\")\nl3 = plot_sol(sol_frozen, colorant\"#332288\")\nl4 = plot_sol(sol_frozen_cg2, colorant\"#CCEE88\")\nl5 = plot_sol(sol_frozen_cg23, colorant\"#88CCEE\")\nl6 = plot_sol(sol_frozen_cg3, colorant\"#44AA99\")\nLegend(f[1, 2],\n    [l1, l2, l3, l4, l5, l6],\n    [\"Lie Euler\", \"RKMK4\", \"Euler\", \"CG2\", \"CG2(3)\", \"CG3\"]\n)\nsave(\"docs/src/assets/img/first_example_solutions.png\", f)","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"And the solutions look like","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"(Image: The ODE solutions)","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"Note that alg_cg23 uses adaptive time stepping.","category":"page"},{"location":"error_estimation.html#Error-estimation","page":"Error estimation","title":"Error estimation","text":"","category":"section"},{"location":"error_estimation.html","page":"Error estimation","title":"Error estimation","text":"Methods with time step adaptation require estimating the error of the solution. The error is then forwarded to algorithms from OrdinaryDiffEq.jl, see documentation.","category":"page"},{"location":"error_estimation.html","page":"Error estimation","title":"Error estimation","text":"Modules = [ManifoldDiffEq]\nPages = [\"error_estimation.jl\"]\nOrder = [:type, :function]","category":"page"},{"location":"error_estimation.html#ManifoldDiffEq.calculate_eest-Tuple{AbstractManifold, Any, Any, Any, Any, Any, Any, Any}","page":"Error estimation","title":"ManifoldDiffEq.calculate_eest","text":"calculate_eest(M::AbstractManifold, utilde, uprev, u, abstol, reltol, internalnorm, t)\n\nEstimate error of a solution of an ODE on manifold M.\n\nArguments\n\nutilde – point on M for error estimation,\nuprev – point from before the current step,\nu – point after the current step`,\nabstol - abolute tolerance,\nreltol - relative tolerance,\ninternalnorm – copied internalnorm from the integrator,\nt – time at which the error is estimated.\n\n\n\n\n\n","category":"method"},{"location":"error_estimation.html#ManifoldDiffEq.reltol_norm-Tuple{AbstractManifold, Any}","page":"Error estimation","title":"ManifoldDiffEq.reltol_norm","text":"reltol_norm(M::AbstractManifold, u)\n\nEstimate the fraction d_{min}/eps(number_eltype(u)) where d_{min} is the distance between u, a point on M, and the nearest distinct point on M representable in the representation of u.\n\n\n\n\n\n","category":"method"},{"location":"error_estimation.html#Literature","page":"Error estimation","title":"Literature","text":"","category":"section"},{"location":"index.html#ManifoldDiffEq","page":"Home","title":"ManifoldDiffEq","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"The package ManifoldDiffEq aims to provide a library of differential equation solvers on manifolds. The library is built on top of Manifolds.jl and follows the interface of OrdinaryDiffEq.jl.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"The code below demonstrates usage of ManifoldDiffEq to solve a simple equation and visualize the results.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Methods implemented in this library are described for example in[HLW10].","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"ManifoldDiffEq.ManifoldODEProblem","category":"page"},{"location":"index.html#ManifoldDiffEq.ManifoldODEProblem","page":"Home","title":"ManifoldDiffEq.ManifoldODEProblem","text":"ManifoldODEProblem\n\nA general problem for ODE problems on Riemannian manifolds.\n\nFields\n\nf the tangent vector field f(u,p,t)\nu0 the initial condition\ntspan time interval for the solution\np constant parameters for f`\nkwargs A callback to be applied to every solver which uses the problem.\nproblem_type type of problem\nmanifold the manifold the vector field is defined on\n\n\n\n\n\n","category":"type"},{"location":"index.html#Literature","page":"Home","title":"Literature","text":"","category":"section"},{"location":"notation.html#Notation","page":"Notation","title":"Notation","text":"","category":"section"},{"location":"notation.html","page":"Notation","title":"Notation","text":"Notation of ManifoldDiffEq.jl mostly follows the notation of Manifolds.jl. There are, however, a few changes to more closely match the notation of the DiffEq ecosystem. Namely:","category":"page"},{"location":"notation.html","page":"Notation","title":"Notation","text":"u is often used to denote points on a manifold.\nTangent vectors are usually denoted by X, Y but some places may use the symbol k.\nParameters of the solved function are denoted either by p or params, depending on the context.","category":"page"}]
}
