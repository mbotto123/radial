# radial: ***R***ecovery-b***a***se***d*** ***I***sotropic ***A***daptation ***L***ibrary

> [!NOTE]
> **This repository is in very early development.**

The Recovery-Based Isotropic Adaptation Library (radial) is a library for problem-independent finite element mesh adaptation, based on the deal.ii library and using Gmsh for mesh generation.

Radial is planned to have the following features, some of which may not be available initially as this repository is in early development:
- Problem-independent
- Works with simplex elements
- Works on 2D and 3D problems
- Can use P1 or P2 Lagrange elements[^1]
- Based on re-meshing with no hanging nodes
- Only allows for isotropic adaptation

[^1]: The limitation to P2 is due to the fact that deal.ii does not currently support simplex elements beyond P2. The methods implemented in this library are capable of handling elements beyond P2,
but these elements are not available in deal.ii yet.

## Examples

There are a few example cases included that demonstrate the usage of radial. The basic finite element machinery used in these examples is largely based on the deal.ii tutorial. Users who are
familiar with deal.ii can use these examples to see how radial is incoporated into a typical deal.ii program.

### Compiling an example

To compile an example, start by creating a build directory and running CMake:

```
mkdir build && cd build
cmake ..
```

Then, run `make <example_name>.cc`. For example, to compile the $L^2$ projection adaptation example:

```
make adapt_projection_2D
```

## Theory

This repository is an open-source implementation of some (but not all) of the methodolody described in the thesis:

> M. Botto Tornielli. "Approximate L<sup>2</sup> Error Control by Solution Post-Processing for Finite Element Solutions of PDEs with Higher-Order Adaptive Methods."
> SM Thesis. Massachusetts Institute of Technology, Center for Computational Science and Engineering, 2025. https://hdl.handle.net/1721.1/164837.

This thesis includes many references on finite element solution and gradient recovery methods, but three particuarly important ones to the code in this repository are highlighted below:

> A. Naga and Z. Zhang. “A Posteriori Error Estimates Based on the Polynomial Preserving Recovery”.
> In: _SIAM J. Numer. Anal._ 42.4 (2004), pp. 1780–1800

> O. Zienkiewicz and J. Zhu. “The superconvergent patch recovery and a posteriori error estimates. Part 1: The recovery technique”.
> In: _International Journal for Numerical Methods in Engineering_ 33 (1992), pp. 1331–1364.

> L. Chamoin and F. Legoll. “An Introductory Review on A Posteriori Error Estimation in Finite Element Computations”.
> In: _SIAM Review_ 65.4 (2023), pp. 963–1028.

One of the mesh size formulas used in this library is from the paper by Chamoin and Legoll.

## Acknowledgements

### deal.ii

While the adaptation methodology implemented in this library is different from the hanging-node approach native to deal.ii, deal.ii is used for all of the fundamental finite element machinery
underlying this library. The deal.ii library is therefore acknowledged with the following references:

> D. Arndt, W. Bangerth, M. Bergbauer, B. Blais, M. Fehling, R. Gassmöller, T. Heister, L. Heltai, M. Kronbichler, M. Maier, P. Munch, S. Scheuerman, B. Turcksin, S. Uzunbajakau, D. Wells, M. Wichrowski.
> "The deal.ii Library, Version 9.7". In: _Journal of Numerical Mathematics_, 33.4 (2025), pp. 403-415.

> D. Arndt, W. Bangerth, D. Davydov, T. Heister, L. Heltai, M. Kronbichler, M. Maier, J.-P. Pelteret, B. Turcksin, D. Wells.
> "The deal.ii finite element library: design, features, and insights". In: _Computers & Mathematics with Applications_, 82 (2021), pp. 407-422.

### Gmsh

Gmsh is used for re-meshing. Gmsh is acknowledged with the following reference:

> C. Geuzaine and J.-F. Remacle. "Gmsh: a three-dimensional finite element mesh generator with built-in pre- and post-processing facilities".
> In: _International Journal for Numerical Methods in Engineering_, 79.11 (2009), pp. 1309-1331.

One of the mesh size formulas used in this library is from the Gmsh example `gmsh/examples/api/adapt_mesh.cpp` developed by C. Geuzaine.
