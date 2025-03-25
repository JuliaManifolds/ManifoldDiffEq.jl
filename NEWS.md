# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] – 2025-03-25

### Changed

* Substituting `SciMLBase.AbstractDiffEqOperator` with `SciMLOperators.AbstractSciMLOperator`.

## [0.2.1] – 2025-02-10

### Changed

* Increased `ManifoldsBase.jl` compatibility to 1.0.

### Fixed

* Added a few missing retraction methods in some solvers.

## [0.2.0] – 2024-09-03

### Changed

* ODE solutions are no longer stored in `SciMLBase.ODESolution` but in `ManifoldODESolution` to avoid having to define `ndims` for all points.
* `SciMLBase.jl` newer than 2.39 is supported again.
* Julia 1.10 is now required due to SciML dependencies.
