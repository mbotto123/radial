#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/table.h>

#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include <solution_recovery.h>

//-------------------------------------------------------------------------//
// Test of solution recovery function in 3D
//-------------------------------------------------------------------------//

using namespace dealii;

void ppr_P1_interpolant_test_3D()
{
  const int dim = 3;
  const int order = 1;

  //-------------------------------------------------------------------------//
  // Function to interpolate from
  const int order_enriched = order + 1;

  const int basis_size = (1.0/6.0) *
                         (order_enriched + 1) *
                         (order_enriched + 2) *
                         (order_enriched + 3);

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

  Functions::Polynomial<dim> quadratic(exponents, coeffs);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Mesh
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, 2);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Base finite element field
  const FE_SimplexP<dim> fe(order);
  MappingP1<dim> mapping;

  DoFHandler<dim> dof_handler(triangulation); 
  dof_handler.distribute_dofs(fe);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Compute linear interpolant
  Vector<double> interpolant(dof_handler.n_dofs());

  VectorTools::interpolate(dof_handler, quadratic, interpolant);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Enriched finite element field
  const FE_SimplexP<dim> fe_enriched(order_enriched);

  DoFHandler<dim> dof_handler_enriched(triangulation); 
  dof_handler_enriched.distribute_dofs(fe_enriched);

  Vector<double> interpolant_enriched(dof_handler_enriched.n_dofs());
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Solution recovery
  radial::recover_solution_ppr(dof_handler, mapping, interpolant,
                               dof_handler_enriched, interpolant_enriched);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Compute error between enriched solution and exact solution
  // (exact solution should be captured exactly by enriched solution)
  VectorTools::NormType norm = VectorTools::L2_norm;

  QGaussSimplex<dim> error_quadrature(order_enriched + 1);

  Vector<double> cell_errors(triangulation.n_cells());

  VectorTools::integrate_difference(mapping,
                                    dof_handler_enriched,
                                    interpolant_enriched,
                                    quadratic,
                                    cell_errors,
                                    error_quadrature,
                                    norm);
  double global_error = VectorTools::compute_global_error(triangulation,
                                                          cell_errors,
                                                          norm);

  if (global_error < 1e-14)
    std::cout << "P1 Passed" << std::endl;
  else
    std::cout << "P1 Failed" << std::endl;
  //-------------------------------------------------------------------------//
}

void ppr_P2_interpolant_test_3D()
{
  const int dim = 3;
  const int order = 2;

  //-------------------------------------------------------------------------//
  // Function to interpolate from
  const int order_enriched = order + 1;

  const int basis_size = (1.0/6.0) *
                         (order_enriched + 1) *
                         (order_enriched + 2) *
                         (order_enriched + 3);

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

  Functions::Polynomial<dim> cubic(exponents, coeffs);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Mesh
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, 2);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Base finite element field
  const FE_SimplexP<dim> fe(order);
  MappingP1<dim> mapping;

  DoFHandler<dim> dof_handler(triangulation); 
  dof_handler.distribute_dofs(fe);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Compute linear interpolant
  Vector<double> interpolant(dof_handler.n_dofs());

  VectorTools::interpolate(dof_handler, cubic, interpolant);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Enriched finite element field
  const FE_SimplexP<dim> fe_enriched(order_enriched);

  DoFHandler<dim> dof_handler_enriched(triangulation); 
  dof_handler_enriched.distribute_dofs(fe_enriched);

  Vector<double> interpolant_enriched(dof_handler_enriched.n_dofs());
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Solution recovery
  radial::recover_solution_ppr(dof_handler, mapping, interpolant,
                               dof_handler_enriched, interpolant_enriched);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Compute error between enriched solution and exact solution
  // (exact solution should be captured exactly by enriched solution)
  VectorTools::NormType norm = VectorTools::L2_norm;

  QGaussSimplex<dim> error_quadrature(order_enriched + 1);

  Vector<double> cell_errors(triangulation.n_cells());

  VectorTools::integrate_difference(mapping,
                                    dof_handler_enriched,
                                    interpolant_enriched,
                                    cubic,
                                    cell_errors,
                                    error_quadrature,
                                    norm);
  double global_error = VectorTools::compute_global_error(triangulation,
                                                          cell_errors,
                                                          norm);

  if (global_error < 1e-14)
    std::cout << "P2 Passed" << std::endl;
  else
    std::cout << "P2 Failed" << std::endl;
  //-------------------------------------------------------------------------//
}

int main()
{
  ppr_P1_interpolant_test_3D();
  // TODO: 3D P2 doesn't pass yet. There is a conditioning issue.
  // Try making the patch larger, using a Legendre basis, or just adding support
  // for integral least-squares.
  ppr_P2_interpolant_test_3D(); // fails!
}
