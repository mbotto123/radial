#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/table.h>

#include <deal.II/lac/vector.h>

#include <iostream>
#include <cmath>

#include <recovery_common.h>

//-------------------------------------------------------------------------//
// Test of patch polynomial basis creation in 3D
//-------------------------------------------------------------------------//

using namespace dealii;

void patch_basis_test_P2_3D()
{
  const int dim = 3;

  // This represents the order of the finite element solution when this function
  // is used in the context of recovery.
  const unsigned int order = 1;

  // In the context of recovery, the patch basis must be one order higher than
  // the finite element solution.
  const unsigned int basis_order = order + 1;

  const unsigned int basis_size = (1.0/6.0) * (basis_order + 1) * (basis_order + 2) * (basis_order + 3);

  std::vector<std::function<double(Point<dim>)>> patch_basis_funcs(basis_size);

  radial::create_patch_basis(order, patch_basis_funcs);

  // Monomial coefficients
  std::vector<double> coeffs(basis_size);
  coeffs = {1, -3, 4, 1, 2, 7, 6, -5, -9, 1};

  // Monomial exponents
  const double exponents_array[] = {0, 0, 0,  // x^0 y^0 z^0
                                    1, 0, 0,  // x^1 y^0 z^0
                                    0, 1, 0,  // x^0 y^1 z^0
                                    0, 0, 1,  // x^0 y^0 z^1
                                    2, 0, 0,  // x^2 y^0 z^0
                                    1, 1, 0,  // x^1 y^1 z^0
                                    1, 0, 1,  // x^1 y^0 z^1
                                    0, 2, 0,  // x^0 y^2 z^0
                                    0, 1, 1,  // x^0 y^1 z^1
                                    0, 0, 2}; // x^0 y^0 z^2
  Table<2, double> exponents(basis_size, dim, exponents_array);

  // Exact function to test against
  Functions::Polynomial<dim> exact_func(exponents, coeffs);

  // Test equivalence at a point
  Point<dim> test_point = {0.5, 0.5, 0.5};

  double exact_val = exact_func.value(test_point);

  double test_val = 0;
  for (unsigned int i = 0; i < basis_size; i++)
    test_val += coeffs[i] * patch_basis_funcs[i](test_point);

  double relative_error = std::abs(exact_val - test_val) / exact_val;

  if (relative_error < 1e-14)
    std::cout << "P2 Passed" << std::endl;
  else
    std::cout << "P2 Failed" << std::endl;
}

void patch_basis_test_P3_3D()
{
  const int dim = 3;

  // This represents the order of the finite element solution when this function
  // is used in the context of recovery.
  const unsigned int order = 2;

  // In the context of recovery, the patch basis must be one order higher than
  // the finite element solution.
  const unsigned int basis_order = order + 1;

  const unsigned int basis_size = (1.0/6.0) * (basis_order + 1) * (basis_order + 2) * (basis_order + 3);

  std::vector<std::function<double(Point<dim>)>> patch_basis_funcs(basis_size);

  radial::create_patch_basis(order, patch_basis_funcs);

  // Monomial coefficients
  std::vector<double> coeffs(basis_size);
  coeffs = {1, -3, 4, 1, 2, 7, 6, -5, -9, 1, 6, 10, 2, -7, 1, 5, -1, 3, -5, 3};

  // Monomial exponents
  const double exponents_array[] = {0, 0, 0,  // x^0 y^0 z^0
                                    1, 0, 0,  // x^1 y^0 z^0
                                    0, 1, 0,  // x^0 y^1 z^0
                                    0, 0, 1,  // x^0 y^0 z^1
                                    2, 0, 0,  // x^2 y^0 z^0
                                    1, 1, 0,  // x^1 y^1 z^0
                                    1, 0, 1,  // x^1 y^0 z^1
                                    0, 2, 0,  // x^0 y^2 z^0
                                    0, 1, 1,  // x^0 y^1 z^1
                                    0, 0, 2,  // x^0 y^0 z^2
                                    3, 0, 0,  // x^3 y^0 z^0
                                    2, 1, 0,  // x^2 y^1 z^0
                                    2, 0, 1,  // x^2 y^0 z^1
                                    1, 2, 0,  // x^1 y^2 z^0
                                    1, 1, 1,  // x^1 y^1 z^1
                                    1, 0, 2,  // x^1 y^0 z^2
                                    0, 3, 0,  // x^0 y^3 z^0
                                    0, 2, 1,  // x^0 y^2 z^1
                                    0, 1, 2,  // x^0 y^1 z^2
                                    0, 0, 3}; // x^0 y^0 z^3
  Table<2, double> exponents(basis_size, dim, exponents_array);

  // Exact function to test against
  Functions::Polynomial<dim> exact_func(exponents, coeffs);

  // Test equivalence at a point
  Point<dim> test_point = {0.5, 0.5, 0.5};

  double exact_val = exact_func.value(test_point);

  double test_val = 0;
  for (unsigned int i = 0; i < basis_size; i++)
    test_val += coeffs[i] * patch_basis_funcs[i](test_point);

  double relative_error = std::abs(exact_val - test_val) / exact_val;

  if (relative_error < 1e-14)
    std::cout << "P3 Passed" << std::endl;
  else
    std::cout << "P3 Failed" << std::endl;
}

int main()
{
  patch_basis_test_P2_3D();
  patch_basis_test_P3_3D();
}