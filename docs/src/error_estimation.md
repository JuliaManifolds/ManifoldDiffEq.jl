# Error estimation

Methods with time step adaptation require estimating the error of the solution. The error
is then forwarded to algorithms from OrdinaryDiffEq.jl, see [documentation](https://diffeq.sciml.ai/stable/extras/timestepping/).

```@autodocs
Modules = [ManifoldDiffEq]
Pages = ["error_estimation.jl"]
Order = [:type, :function]
```

## Literature
