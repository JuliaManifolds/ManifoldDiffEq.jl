# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.6] - unreleased

### Changed

* `CompatHelper` bumped to v3.

## [0.2.5] - 2025-10-18

### Added

* Compatibility with SciMLOperators v0.4

### Fixed

* Example usage was updated to the `LieGroups.jl` refactor.

## [0.2.4] - 2025-10-09

### Changed

* Bumped dependencies of all JuliaManifolds ecosystem packages to be consistent with ManifoldsBase.jl 2.0 and Manifolds.jl 0.11
* `ManifoldLieEuler` and `RKMK4` no longer store the manifold separately from the action, and instead use the manifold used in the action.

## [0.2.3] - 2025-05-11

### Changed

* Enable ManifoldODESolution to support abstract numeric types (include Complex with previously supported Real)

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
