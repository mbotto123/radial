#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>

#include <iostream>
#include <cmath>

#include <recovery_common.h>

//-------------------------------------------------------------------------//
// Test of patch polynomial basis creation in 2D
//-------------------------------------------------------------------------//

using namespace dealii;

template <int dim>
class Quadratic : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template<int dim>
double Quadratic<dim>::value(const Point<dim> &p,
                            const unsigned int /*component*/) const
{
  if (dim == 2)
  {
    double x = p[0];
    double y = p[1];

    double linear = 1 - 3*x + 4*y;
    double quadratic = 2*x*x  + 7*x*y - 5*y*y;

    return linear + quadratic;
  }
  else if (dim == 3)
  {
    double x = p[0];
    double y = p[1];
    double z = p[2];

    double linear = 1 - 3*x + 4*y + z;
    double quadratic = 2*x*x + 7*x*y + 6*x*z - 5*y*y - 9*y*z + z*z;

    return linear + quadratic;
  }
}

template <int dim>
class Cubic : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template<int dim>
double Cubic<dim>::value(const Point<dim> &p,
                            const unsigned int /*component*/) const
{
  if (dim == 2)
  {
    double x = p[0];
    double y = p[1];

    double linear = 1 - 3*x + 4*y;
    double quadratic = 2*x*x  + 7*x*y - 5*y*y;
    double cubic_x = 6*x*x*x + 10*x*x*y - 7*x*y*y;
    double cubic_y = -y*y*y;

    return linear + quadratic + cubic_x + cubic_y;
  }
  else if (dim == 3)
  {
    double x = p[0];
    double y = p[1];
    double z = p[2];

    double linear = 1 - 3*x + 4*y + z;
    double quadratic = 2*x*x + 7*x*y + 6*x*z - 5*y*y - 9*y*z + z*z;
    double cubic_x = 6*x*x*x + 10*x*x*y + 2*x*x*z - 7*x*y*y  + x*y*z + 5*x*z*z;
    double cubic_y = -y*y*y + 3*y*y*z;
    double cubic_z = -5*y*z*z + 3*z*z*z;

    return linear + quadratic + cubic_x + cubic_y + cubic_z;
  }
}

void patch_basis_test_P2_2D()
{
  const int dim = 2;

  // This represents the order of the finite element solution when this function
  // is used in the context of recovery.
  const unsigned int order = 1;

  // In the context of recovery, the patch basis must be one order higher than
  // the finite element solution.
  const unsigned int basis_order = order + 1;

  const unsigned int basis_size = 0.5 * (basis_order + 1) * (basis_order + 2);

  std::vector<std::function<double(Point<dim>)>> patch_basis_funcs(basis_size);

  radial::create_patch_basis(order, patch_basis_funcs);

  // Coefficients to match the exact function we're testing against
  Vector<double> coeffs(basis_size);
  coeffs(0) =  1;
  coeffs(1) = -3;
  coeffs(2) =  4;
  coeffs(3) =  2;
  coeffs(4) =  7;
  coeffs(5) = -5;

  Quadratic<dim> exact_func;

  // Test equivalence at a point
  Point<dim> test_point = {0.5, 0.5};

  double exact_val = exact_func.value(test_point);

  double test_val = 0;
  for (unsigned int i = 0; i < basis_size; i++)
    test_val += coeffs(i) * patch_basis_funcs[i](test_point);

  double relative_error = std::abs(exact_val - test_val) / exact_val;

  if (relative_error < 1e-14)
    std::cout << "P2 Passed" << std::endl;
  else
    std::cout << "P2 Failed" << std::endl;
}

void patch_basis_test_P3_2D()
{
  const int dim = 2;

  // This represents the order of the finite element solution when this function
  // is used in the context of recovery.
  const unsigned int order = 2;

  // In the context of recovery, the patch basis must be one order higher than
  // the finite element solution.
  const unsigned int basis_order = order + 1;

  const unsigned int basis_size = 0.5 * (basis_order + 1) * (basis_order + 2);

  std::vector<std::function<double(Point<dim>)>> patch_basis_funcs(basis_size);

  radial::create_patch_basis(order, patch_basis_funcs);

  // Coefficients to match the exact function we're testing against
  Vector<double> coeffs(basis_size);
  coeffs(0) =  1;
  coeffs(1) = -3;
  coeffs(2) =  4;
  coeffs(3) =  2;
  coeffs(4) =  7;
  coeffs(5) = -5;
  coeffs(6) =  6;
  coeffs(7) = 10;
  coeffs(8) = -7;
  coeffs(9) = -1;

  Cubic<dim> exact_func;

  // Test equivalence at a point
  Point<dim> test_point = {0.5, 0.5};

  double exact_val = exact_func.value(test_point);

  double test_val = 0;
  for (unsigned int i = 0; i < basis_size; i++)
    test_val += coeffs(i) * patch_basis_funcs[i](test_point);

  double relative_error = std::abs(exact_val - test_val) / exact_val;

  if (relative_error < 1e-14)
    std::cout << "P3 Passed" << std::endl;
  else
    std::cout << "P3 Failed" << std::endl;
}

int main()
{
  patch_basis_test_P2_2D();
  patch_basis_test_P3_2D();
}