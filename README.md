# osqp-cpp: A C++ wrapper for [OSQP](https://osqp.org/)

A C++ wrapper for [OSQP](https://github.com/oxfordcontrol/osqp), an
[ADMM](http://stanford.edu/~boyd/admm.html)-based solver for
[quadratic programming](https://en.wikipedia.org/wiki/Quadratic_programming).

Compared with OSQP's native C interface, the wrapper provides a more convenient
input format using Eigen sparse matrices and handles the lifetime of the
`OSQPWorkspace` struct. This package has similar functionality to
[osqp-eigen](https://github.com/robotology/osqp-eigen).

The full API is documented in-line in `osqp++.h`. We describe only the input
format in this README.

Note: OSQP uses looser default tolerances than other similar solvers. We
recommend looking at the description of the convergence tolerances in Section
3.4 of the OSQP [paper](https://arxiv.org/abs/1711.08013) and adjusting
tolerances via the `OsqpSettings` struct as appropriate.

This is not an officially supported Google product.

## `OsqpInstance` format

OSQP solves the convex quadratic optimization problem:

$$
\begin{align}
\min_x \quad& \frac{1}{2} x^TPx + q^Tx \\
\text{s.t.}\quad & l \le Ax \le u
\end{align}
$$

where $$P$$ is a symmetric positive semi-definite matrix.

The inequalities are component-wise, and equalities may be enforced by setting
$$l_i = u_i$$ for some row $$i$$. Single-sided inequalities can be enforced by
setting the lower or upper bounds to negative or positive infinity
(`std::numeric_limits<double>::infinity()`), respectively.

This maps to the `OsqpInstance` struct in `osqp.h` as follows.

-   `objective_matrix` is $$P$$.
-   `objective_vector` is $$q$$.
-   `constraint_matrix` is $$A$$.
-   `lower_bounds` is $$l$$.
-   `upper_bounds` is $$u$$.

## Example usage

The code below formulates and solves the following 2-dimensional optimization
problem:

$$
\begin{align} \min_{x,y} \quad& x^2 + (1/2)xy + y^2 + x \\
\text{s.t.}\quad& x \ge 1. \end{align}
$$

```C++
const double kInfinity = std::numeric_limits<double>::infinity();
SparseMatrix<double> objective_matrix(2, 2);
const Triplet<double> kTripletsP[] = {
    {0, 0, 2.0}, {1, 0, 0.5}, {0, 1, 0.5}, {1, 1, 2.0}};
objective_matrix.setFromTriplets(std::begin(kTripletsP),
                                   std::end(kTripletsP));

SparseMatrix<double> constraint_matrix(1, 2);
const Triplet<double> kTripletsA[] = {{0, 0, 1.0}};
constraint_matrix.setFromTriplets(std::begin(kTripletsA),
                                      std::end(kTripletsA));

OsqpInstance instance;
instance.objective_matrix = objective_matrix;
instance.objective_vector.resize(2);
instance.objective_vector << 1.0, 0.0;
instance.constraint_matrix = constraint_matrix;
instance.lower_bounds.resize(1);
instance.lower_bounds << 1.0;
instance.upper_bounds.resize(1);
instance.upper_bounds << kInfinity;

OsqpSolver solver;
OsqpSettings settings;
// Edit settings if appropriate.
auto status = solver.Init(instance, settings);
// Assuming status.ok().
OsqpExitCode exit_code = solver.Solve();
// Assuming exit_code == OsqpExitCode::kOptimal.
double optimal_objective = solver.objective_value();
Eigen::VectorXd optimal_solution = solver.primal_solution();
```

## Installation

osqp++ requires CMake, a C++17 compiler, and the following packages:

- [OSQP](https://github.com/oxfordcontrol/osqp)
- [abseil-cpp](https://github.com/abseil/abseil-cpp)
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [googletest](https://github.com/google/googletest) (for testing only)

On Debian/Ubuntu systems you may install Eigen via the `libeigen3-dev` package.
Then run the following:

```sh
$ git clone https://github.com/abseil/abseil-cpp.git
$ git clone --recursive https://github.com/oxfordcontrol/osqp.git
$ git clone https://github.com/google/googletest.git
$ mkdir build; cd build
$ cmake ..
$ make
$ make test
```

So far, the interface has been tested only on Linux. Contributions to support
additional platforms are welcome.

## FAQ

-   Is OSQP deterministic?
    -   No, not in its default configuration. Section 5.2 of the OSQP
        [paper](https://arxiv.org/abs/1711.08013) describes that the update rule
        for $$\rho$$ depends on the ratio between the runtime of the iterations
        and the runtime of the numerical factorization. Setting `adaptive_rho`
        to `false` disables this update rule and makes OSQP deterministic, but
        this could significantly slow down OSQP's convergence.
