#include <iterator>
#include <limits>
#include <iostream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "Eigen/SparseCore"
#include "osqp++.h"

using namespace osqp;
using ::Eigen::SparseMatrix;
using ::Eigen::Triplet;

constexpr double kTolerance = 1.0e-5;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

int main()
{
    // Minimize -x subject to:
  // x + y <= 1, x >= 0, y >= 0.

    SparseMatrix<double> constraint_matrix(3, 2);
    const Triplet<double> kTripletsA[] = {
        {0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {2, 1, 1.0} };
    constraint_matrix.setFromTriplets(std::begin(kTripletsA),
        std::end(kTripletsA));

    OsqpInstance instance;
    instance.objective_matrix = SparseMatrix<double>(2, 2);
    instance.objective_vector.resize(2);
    instance.objective_vector << -1.0, 0.0;
    instance.constraint_matrix = constraint_matrix;
    instance.lower_bounds.resize(3);
    instance.lower_bounds << -kInfinity, 0.0, 0.0;
    instance.upper_bounds.resize(3);
    instance.upper_bounds << 1.0, kInfinity, kInfinity;

    OsqpSolver solver;
    OsqpSettings settings;
    // absolute_convergence_tolerance (eps_abs) is an l_2 tolerance on the
    // residual vector, so this is safe given we use kTolerance
    // as an l_infty tolerance.
    settings.eps_abs = kTolerance;
    settings.eps_rel = 0.0;
    solver.IsInitialized();
    solver.Init(instance, settings).ok();
    solver.IsInitialized();
    solver.Solve();

    std::cout << "Value: " << solver.objective_value() << "\n";
}