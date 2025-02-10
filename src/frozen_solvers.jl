
"""
    ManifoldEuler

The manifold Euler algorithm for problems in the [`ExplicitManifoldODEProblemType`](@ref)
formulation.
"""
struct ManifoldEuler{TM<:AbstractManifold,TR<:AbstractRetractionMethod} <:
       AbstractManifoldDiffEqAlgorithm
    manifold::TM
    retraction_method::TR
end

alg_order(::ManifoldEuler) = 1

"""
    ManifoldEulerCache

Mutable cache for [`ManifoldEuler`](@ref).
"""
struct ManifoldEulerCache <: OrdinaryDiffEqMutableCache end

"""
    ManifoldEulerConstantCache

Cache for [`ManifoldEuler`](@ref).
"""
struct ManifoldEulerConstantCache <: OrdinaryDiffEqConstantCache end

function alg_cache(
    alg::ManifoldEuler,
    u,
    rate_prototype,
    uEltypeNoUnits,
    uBottomEltypeNoUnits,
    tTypeNoUnits,
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{true},
)
    return ManifoldEulerCache()
end

function perform_step!(integrator, ::ManifoldEulerCache, repeat_step = false)
    u = integrator.u
    t = integrator.t
    alg = integrator.alg

    integrator.k[1] = integrator.f(u, integrator.p, t)
    retract_fused!(
        alg.manifold,
        u,
        u,
        integrator.k[1],
        integrator.dt,
        alg.retraction_method,
    )

    return integrator.stats.nf += 1
end

function initialize!(integrator, ::ManifoldEulerCache)
    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
    integrator.stats.nf += 1
    integrator.kshortsize = 1
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero.(integrator.fsalfirst)
    return integrator.k[1] = integrator.fsalfirst
end


@doc raw"""
    CG2

A Crouch-Grossmann algorithm of second order for problems in the
[`ExplicitManifoldODEProblemType`](@ref) formulation.
The Butcher tableau is identical to the Euclidean RK2:

```math
\begin{array}{c|cc}
0 & 0 \\
\frac{1}{2} & \frac{1}{2} & 0 \\
\hline
& 0 & 1
\end{array}
```
"""
struct CG2{TM<:AbstractManifold,TR<:AbstractRetractionMethod} <:
       AbstractManifoldDiffEqAlgorithm
    manifold::TM
    retraction_method::TR
end

alg_order(::CG2) = 2

"""
    CG2Cache

Mutable cache for [`CG2`](@ref).
"""
struct CG2Cache{TX,TK2u} <: OrdinaryDiffEqMutableCache
    X1::TX
    X2u::TK2u
    X2::TX
end

function alg_cache(
    alg::CG2,
    u,
    rate_prototype,
    uEltypeNoUnits,
    uBottomEltypeNoUnits,
    tTypeNoUnits,
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{true},
)
    return CG2Cache(allocate(rate_prototype), allocate(u), allocate(rate_prototype))
end

function initialize!(integrator, cache::CG2Cache)
    integrator.kshortsize = 2
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t)
    integrator.stats.nf += 1

    integrator.fsallast = zero.(integrator.fsalfirst)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    return nothing
end

function perform_step!(integrator, cache::CG2Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p, alg = integrator

    M = alg.manifold
    f(cache.X1, u, p, t)
    dt2 = dt / 2
    retract_fused!(M, cache.X2u, u, cache.X1, dt2, alg.retraction_method)
    f(cache.X2, cache.X2u, p, t + dt2)
    k2t = f.f.operator_vector_transport(M, cache.X2u, cache.X2, u, p, t + dt2, t)
    retract_fused!(M, u, u, k2t, dt, alg.retraction_method)

    return integrator.stats.nf += 2
end



@doc raw"""
    CG2_3

A Crouch-Grossmann algorithm of order 2(3) for problems in the
[`ExplicitManifoldODEProblemType`](@ref) formulation.
The Butcher tableau reads (see tableau (5) of [EngøMarthinsen:1998](@cite)):

```math
\begin{array}{c|ccc}
0 & 0 \\
\frac{3}{4} & \frac{3}{4} & 0 \\
\frac{17}{24} & \frac{119}{216} & \frac{17}{108} & 0\\
\hline
& \frac{3}{4} & \frac{31}{4} & \frac{-15}{2}
& \frac{13}{51} & -\frac{2}{3} & \frac{24}{17}
\end{array}
```
The last row is used for error estimation.
"""
struct CG2_3{TM<:AbstractManifold,TR<:AbstractRetractionMethod} <:
       AbstractManifoldDiffEqAdaptiveAlgorithm
    manifold::TM
    retraction_method::TR
end

alg_order(::CG2_3) = 2

"""
    CG2_3Cache

Cache for [`CG2_3`](@ref).
"""
struct CG2_3Cache{TX,TP} <: OrdinaryDiffEqMutableCache
    X1::TX
    X2::TX
    X3::TX
    X2u::TP
    X3u::TP
    uhat::TP
end

function alg_cache(
    alg::CG2_3,
    u,
    rate_prototype,
    uEltypeNoUnits,
    uBottomEltypeNoUnits,
    tTypeNoUnits,
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{true},
)
    return CG2_3Cache(
        allocate(rate_prototype),
        allocate(rate_prototype),
        allocate(rate_prototype),
        allocate(u),
        allocate(u),
        allocate(u),
    )
end

function DiffEqBase.get_tmp_cache(integrator, ::CG2_3, cache::CG2_3Cache)
    return (cache.X1,)
end

function initialize!(integrator, cache::CG2_3Cache)
    integrator.kshortsize = 2
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t)
    integrator.stats.nf += 1

    integrator.fsallast = zero.(integrator.fsalfirst)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    return nothing
end

function perform_step!(integrator, cache::CG2_3Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p, alg = integrator
    M = alg.manifold

    f(cache.X1, u, p, t)
    c2h = (3 // 4) * dt
    c3h = (17 // 24) * dt
    a21h = (3 // 4) * dt
    a31h = (119 // 216) * dt
    a32h = (17 // 108) * dt
    b1 = (3 // 4) * dt
    b2 = (31 // 4) * dt
    b3 = (-15 // 2) * dt
    b1hat = (13 // 51) * dt
    b2hat = (-2 // 3) * dt
    b3hat = (24 // 17) * dt
    retract_fused!(M, cache.X2u, u, cache.X1, a21h, alg.retraction_method)
    f(cache.X2, cache.X2u, p, t + c2h)
    retract_fused!(M, cache.X3u, u, cache.X1, a31h, alg.retraction_method)
    k2tk3u = f.f.operator_vector_transport(M, cache.X2u, cache.X2, cache.X3u, p, t, t + c2h)
    retract_fused!(M, cache.X3u, cache.X3u, k2tk3u, a32h, alg.retraction_method)
    f(cache.X3, cache.X3u, p, t + c3h)
    if integrator.opts.adaptive
        copyto!(M, cache.uhat, u)
    end

    retract_fused!(M, u, u, cache.X1, b1, alg.retraction_method)
    X2tu = f.f.operator_vector_transport(M, cache.X2u, cache.X2, u, p, t + c2h, t)
    retract_fused!(M, u, u, X2tu, b2, alg.retraction_method)
    X3tu = f.f.operator_vector_transport(M, cache.X3u, cache.X3, u, p, t + c3h, t)
    retract_fused!(M, u, u, X3tu, b3, alg.retraction_method)

    if integrator.opts.adaptive
        uhat = cache.uhat
        retract_fused!(M, uhat, uhat, cache.X1, b1hat, alg.retraction_method)
        X2tu = f.f.operator_vector_transport(M, cache.X2u, cache.X2, uhat, p, t + c2h, t)
        retract_fused!(M, uhat, uhat, X2tu, b2hat, alg.retraction_method)
        X3tu = f.f.operator_vector_transport(M, cache.X3u, cache.X3, uhat, p, t + c3h, t)
        retract_fused!(M, uhat, uhat, X3tu, b3hat, alg.retraction_method)

        integrator.EEst = calculate_eest(
            M,
            uhat,
            uprev,
            u,
            integrator.opts.abstol,
            integrator.opts.reltol,
            integrator.opts.internalnorm,
            t,
        )
    end

    return integrator.stats.nf += 3
end



@doc raw"""
    CG3

A Crouch-Grossmann algorithm of second order for problems in the
[`ExplicitManifoldODEProblemType`](@ref) formulation. See tableau 6.1 of [OwrenMarthinsen:1999](@cite):

```math
\begin{array}{c|ccc}
0 & 0 \\
\frac{3}{4} & \frac{3}{4} & 0 \\
\frac{17}{24} & \frac{119}{216} & \frac{17}{108} & 0\\
\hline
& \frac{13}{51} & -\frac{2}{3} & \frac{24}{17}
\end{array}
```

"""
struct CG3{TM<:AbstractManifold,TR<:AbstractRetractionMethod} <:
       AbstractManifoldDiffEqAlgorithm
    manifold::TM
    retraction_method::TR
end

alg_order(::CG3) = 3

"""
    CG3Cache

Mutable cache for [`CG3`](@ref).
"""
struct CG3Cache{TX,TP} <: OrdinaryDiffEqMutableCache
    X1::TX
    X2::TX
    X3::TX
    X2u::TP
    X3u::TP
end


function alg_cache(
    alg::CG3,
    u,
    rate_prototype,
    uEltypeNoUnits,
    uBottomEltypeNoUnits,
    tTypeNoUnits,
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{true},
)
    return CG3Cache(
        allocate(rate_prototype),
        allocate(rate_prototype),
        allocate(rate_prototype),
        allocate(u),
        allocate(u),
    )
end

function perform_step!(integrator, cache::CG3Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p, alg = integrator
    M = alg.manifold

    f(cache.X1, u, p, t)
    c2h = (3 // 4) * dt
    c3h = (17 // 24) * dt
    a21h = (3 // 4) * dt
    a31h = (119 // 216) * dt
    a32h = (17 // 108) * dt
    b1 = (13 // 51) * dt
    b2 = (-2 // 3) * dt
    b3 = (24 // 17) * dt
    retract_fused!(M, cache.X2u, u, cache.X1, a21h, alg.retraction_method)
    f(cache.X2, cache.X2u, p, t + c2h)
    retract_fused!(M, cache.X3u, u, cache.X1, a31h, alg.retraction_method)
    k2tk3u = f.f.operator_vector_transport(M, cache.X2u, cache.X2, cache.X3u, p, t, t + c2h)
    retract_fused!(M, cache.X3u, cache.X3u, k2tk3u, a32h, alg.retraction_method)
    f(cache.X3, cache.X3u, p, t + c3h)

    retract_fused!(M, u, u, cache.X1, b1, alg.retraction_method)
    X2tu = f.f.operator_vector_transport(M, cache.X2u, cache.X2, u, p, t + c2h, t)
    retract_fused!(M, u, u, X2tu, b2, alg.retraction_method)
    X3tu = f.f.operator_vector_transport(M, cache.X3u, cache.X3, u, p, t + c3h, t)
    retract_fused!(M, u, u, X3tu, b3, alg.retraction_method)

    return integrator.stats.nf += 3
end

function initialize!(integrator, ::CG3Cache)
    integrator.kshortsize = 2
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t)
    integrator.stats.nf += 1

    integrator.fsallast = zero.(integrator.fsalfirst)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    return nothing
end



@doc raw"""
    CG4a

A Crouch-Grossmann algorithm of second order for problems in the
[`ExplicitManifoldODEProblemType`](@ref) formulation. See coefficients from
Example 1 of [JackiewiczMarthinsenOwren:2000](@cite).
"""
struct CG4a{TM<:AbstractManifold,TR<:AbstractRetractionMethod} <:
       AbstractManifoldDiffEqAlgorithm
    manifold::TM
    retraction_method::TR
end

alg_order(::CG4a) = 4

"""
    CG4aCache

Mutable cache for [`CG4a`](@ref).
"""
struct CG4aCache{TX,TP} <: OrdinaryDiffEqMutableCache
    X1::TX
    X2::TX
    X3::TX
    X4::TX
    X5::TX
    X2u::TP
    X3u::TP
    X4u::TP
    X5u::TP
end


function alg_cache(
    alg::CG4a,
    u,
    rate_prototype,
    uEltypeNoUnits,
    uBottomEltypeNoUnits,
    tTypeNoUnits,
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{true},
)
    return CG4aCache(
        allocate(rate_prototype),
        allocate(rate_prototype),
        allocate(rate_prototype),
        allocate(rate_prototype),
        allocate(rate_prototype),
        allocate(u),
        allocate(u),
        allocate(u),
        allocate(u),
    )
end

function perform_step!(integrator, cache::CG4aCache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p, alg = integrator
    M = alg.manifold

    Tdt = typeof(t)

    f(cache.X1, u, p, t)
    c2h = Tdt(0.8177227988124852) * dt
    c3h = Tdt(0.3859740639032449) * dt
    c4h = Tdt(0.3242290522866937) * dt
    c5h = Tdt(0.8768903263420429) * dt
    a21h = Tdt(0.8177227988124852) * dt
    a31h = Tdt(0.3199876375476427) * dt
    a41h = Tdt(0.9214417194464946) * dt
    a51h = Tdt(0.3552358559023322) * dt
    a32h = Tdt(0.0659864263556022) * dt
    a42h = Tdt(0.4997857776773573) * dt
    a52h = Tdt(0.2390958372307326) * dt
    a43h = Tdt(-1.0969984448371582) * dt
    a53h = Tdt(1.3918565724203246) * dt
    a54h = Tdt(-1.1092979392113465) * dt
    b1 = Tdt(0.1370831520630755) * dt
    b2 = Tdt(-0.0183698531564020) * dt
    b3 = Tdt(0.7397813985370780) * dt
    b4 = Tdt(-0.1907142565505889) * dt
    b5 = Tdt(0.3322195591068374) * dt

    retract_fused!(M, cache.X2u, u, cache.X1, a21h, alg.retraction_method)
    f(cache.X2, cache.X2u, p, t + c2h)

    retract_fused!(M, cache.X3u, u, cache.X1, a31h, alg.retraction_method)
    k2tk3u = f.f.operator_vector_transport(M, cache.X2u, cache.X2, cache.X3u, p, t, t + c2h)
    retract_fused!(M, cache.X3u, cache.X3u, k2tk3u, a32h, alg.retraction_method)
    f(cache.X3, cache.X3u, p, t + c3h)

    retract_fused!(M, cache.X4u, u, cache.X1, a41h, alg.retraction_method)
    k2tk4u = f.f.operator_vector_transport(
        M,
        cache.X2u,
        cache.X2,
        cache.X4u,
        p,
        t + c2h,
        t + c4h,
    )
    retract_fused!(M, cache.X4u, cache.X4u, k2tk4u, a42h, alg.retraction_method)
    k3tk4u = f.f.operator_vector_transport(
        M,
        cache.X3u,
        cache.X3,
        cache.X4u,
        p,
        t + c3h,
        t + c4h,
    )
    retract_fused!(M, cache.X4u, cache.X4u, k3tk4u, a43h, alg.retraction_method)
    f(cache.X4, cache.X4u, p, t + c4h)

    retract_fused!(M, cache.X5u, u, cache.X1, a51h, alg.retraction_method)
    k2tk5u = f.f.operator_vector_transport(
        M,
        cache.X2u,
        cache.X2,
        cache.X5u,
        p,
        t + c2h,
        t + c5h,
    )
    retract_fused!(M, cache.X5u, cache.X5u, k2tk5u, a52h, alg.retraction_method)
    k3tk5u = f.f.operator_vector_transport(
        M,
        cache.X3u,
        cache.X3,
        cache.X5u,
        p,
        t + c3h,
        t + c5h,
    )
    retract_fused!(M, cache.X5u, cache.X5u, k3tk5u, a53h, alg.retraction_method)
    k4tk5u = f.f.operator_vector_transport(
        M,
        cache.X4u,
        cache.X4,
        cache.X5u,
        p,
        t + c4h,
        t + c5h,
    )
    retract_fused!(M, cache.X5u, cache.X5u, k4tk5u, a54h, alg.retraction_method)
    f(cache.X5, cache.X5u, p, t + c5h)

    retract_fused!(M, u, u, cache.X1, b1, alg.retraction_method)
    X2tu = f.f.operator_vector_transport(M, cache.X2u, cache.X2, u, p, t + c2h, t)
    retract_fused!(M, u, u, X2tu, b2, alg.retraction_method)
    X3tu = f.f.operator_vector_transport(M, cache.X3u, cache.X3, u, p, t + c3h, t)
    retract_fused!(M, u, u, X3tu, b3, alg.retraction_method)
    X4tu = f.f.operator_vector_transport(M, cache.X4u, cache.X4, u, p, t + c4h, t)
    retract_fused!(M, u, u, X4tu, b4, alg.retraction_method)
    X5tu = f.f.operator_vector_transport(M, cache.X5u, cache.X5, u, p, t + c5h, t)
    retract_fused!(M, u, u, X5tu, b5, alg.retraction_method)

    return integrator.stats.nf += 5
end

function initialize!(integrator, ::CG4aCache)
    integrator.kshortsize = 2
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t)
    integrator.stats.nf += 1

    integrator.fsallast = zero.(integrator.fsalfirst)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    return nothing
end
