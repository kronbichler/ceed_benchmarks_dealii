# ceed_benchmarks_dealii

This project provides small programs to evaluate the ceed benchmark cases
http://ceed.exascaleproject.org/bps with the matrix-free evaluation routines
provided by the deal.II finite element library,
https://github.com/dealii/dealii

### Results for mass operator



### Prerequisites and installation

The benchmark problems are designed as small programs that run against a
compiled deal.II finite element library that in turn needs a C++ compiler
adhering to the C++11 standard, an MPI implementation, and cmake. The
following software packages are needed:

* deal.II, using at least version 8.5.0, see www.dealii.org. deal.II must be
  configured to also include the following external packages (no direct access
  to this packages is necessary, except for the interface through deal.II):

* MPI

* p4est for providing parallel adaptive mesh management on forests of
  quad-trees (2D) or oct-trees (3D). For obtaining p4est, see
  http://www.p4est.org. p4est of at least version 0.3.4.2 is needed for
  adaflo. Installation of p4est can be done via a script provided by deal.II:
```
/path/to/dealii/doc/external-libs/p4est-setup.sh p4est-1.1.tar.gz /path/to/p4est/install
```
  (the last argument specifies the desired installation directory for p4est,
  e.g. $HOME/sw/p4est).

Given these dependencies, the configuration of deal.II can be done
through the following script:
```
cmake \
    -D CMAKE_CXX_FLAGS="-march=native" \
    -D CMAKE_INSTALL_PREFIX="/path/to/dealii/install/" \
    -D DEAL_II_WITH_MPI="ON" \
    -D DEAL_II_WITH_LAPACK="ON" \
    -D DEAL_II_WITH_P4EST="ON" \
    -D P4EST_DIR="/path/to/p4est/install/" \
    ../deal.II
```

Since the matrix-free algorithms in deal.II make intensive use of advanced
processor instruction sets (e.g. vectorization through AVX or similar), it is
recommended to enable processor-specific optimizations (second line,
`-march=native`). The path on the third line specifies the desired
installation directory of deal.II, and the last line points to the location of
the source code of deal.II relative to the folder where the cmake script is
run. After configuration, run

```
make -j8
make install
```

to compile deal.II and install it in the given directory. After installation,
the deal.II source and build folder are no longer necessary (unless you find
bugs in deal.II and need to modify that code). It is also possible to build
the test cases against a build folder of deal.II (which is what the author of
this package does almost exclusively).
