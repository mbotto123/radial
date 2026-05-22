#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
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

#include <recovery_common.h>

//-------------------------------------------------------------------------//
// Test of solving a least-squares problem on a patch in 3D
//-------------------------------------------------------------------------//

using namespace dealii;

void least_squares_patch_test_P1_3D()
{
  const int dim = 3;
  const int order = 1;

  //-------------------------------------------------------------------------//
  // Function to interpolate from
  const int order_enriched = order + 1;

  const int basis_size = radial::get_min_points<dim>(order_enriched);

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

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Reference coordinates of the Lagrange nodes
  // (also used to get the values of the base finite element field at those nodes)
  Quadrature<dim> lagrange_nodes(fe.get_unit_support_points());
  FEValues<dim> fe_values_nodes(mapping,
                                fe,
                                lagrange_nodes,
                                update_values | update_quadrature_points);
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
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Create monomial basis for least-squares fit

  std::vector<std::function<double(Point<dim>)>> patch_basis_funcs(basis_size);
  radial::create_patch_basis(order, patch_basis_funcs);
  //-------------------------------------------------------------------------//

  std::vector<std::set<radial::cell_pointer<dim>>> vertex_to_cell;
  std::vector<std::set<radial::cell_pointer<dim>>> vertex_to_cell_enriched;

  radial::create_vertex_to_cell(dof_handler, dof_handler_enriched,
                                vertex_to_cell, vertex_to_cell_enriched);

  std::set<types::global_dof_index> patch_dofs;

  // Solve least-squares problem for central vertex patch. This patch does not
  // require any growth iterations.
  int v = 13;
  std::set<radial::cell_pointer<dim>> patch_cells = vertex_to_cell[v];
  for (const auto &cell: vertex_to_cell[v])
  {
    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i : fe_values_nodes.dof_indices())
      patch_dofs.insert(local_dof_indices[i]);
  }

  // Vector of least-squares coefficients
  Vector<double> a(basis_size);

  // Reciprocal condition number of the least-squares system on the patch
  double rcond;

  // The reciprocal condition number value at which we consider the
  // least-squares system to be too ill-conditioned to attempt solving.
  double rcond_tol = std::numeric_limits<double>::epsilon();

  radial::solve_least_squares_patch(patch_cells, patch_dofs, patch_basis_funcs,
                                    interpolant, basis_size,
                                    rcond_tol, fe_values_nodes, local_dof_indices,
                                    a, rcond);

  // Test equivalence at a point
  Point<dim> test_point = {0.5, 0.5, 0.5};
  // Since the least-squares problem is solved with normalized coordinates,
  // use the normalized coordinates when passing the point in to the patch
  // basis functions.
  Point<dim> test_point_normalized = {0.0, 0.0, 0.0};

  double exact_val = quadratic.value(test_point);
  std::cout << "P" << order <<  " exact value = " << exact_val << std::endl;

  double test_val = 0;
  for (int i = 0; i < basis_size; i++)
    test_val += a(i) * patch_basis_funcs[i](test_point_normalized);
  std::cout << "P" << order <<  " test value = " << test_val << std::endl;

  double relative_error = std::abs(exact_val - test_val) / exact_val;

  if (relative_error < 1e-14)
    std::cout << "P" << order << " Passed" << std::endl;
  else
    std::cout << "P" << order << " Failed" << std::endl;
}

void least_squares_patch_test_P2_3D()
{
  const int dim = 3;
  const int order = 2;

  //-------------------------------------------------------------------------//
  // Function to interpolate from
  const int order_enriched = order + 1;

  const int basis_size = radial::get_min_points<dim>(order_enriched);

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

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Reference coordinates of the Lagrange nodes
  // (also used to get the values of the base finite element field at those nodes)
  Quadrature<dim> lagrange_nodes(fe.get_unit_support_points());
  FEValues<dim> fe_values_nodes(mapping,
                                fe,
                                lagrange_nodes,
                                update_values | update_quadrature_points);
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
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Create monomial basis for least-squares fit

  std::vector<std::function<double(Point<dim>)>> patch_basis_funcs(basis_size);
  radial::create_patch_basis(order, patch_basis_funcs);
  //-------------------------------------------------------------------------//

  std::vector<std::set<radial::cell_pointer<dim>>> vertex_to_cell;
  std::vector<std::set<radial::cell_pointer<dim>>> vertex_to_cell_enriched;

  radial::create_vertex_to_cell(dof_handler, dof_handler_enriched,
                                vertex_to_cell, vertex_to_cell_enriched);

  std::set<types::global_dof_index> patch_dofs;

  // Solve least-squares problem for central vertex patch. This patch does not
  // require any growth iterations.
  int v = 13;
  std::set<radial::cell_pointer<dim>> patch_cells = vertex_to_cell[v];
  for (const auto &cell: vertex_to_cell[v])
  {
    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i : fe_values_nodes.dof_indices())
      patch_dofs.insert(local_dof_indices[i]);
  }

  // Vector of least-squares coefficients
  Vector<double> a(basis_size);

  // Reciprocal condition number of the least-squares system on the patch
  double rcond;

  // The reciprocal condition number value at which we consider the
  // least-squares system to be too ill-conditioned to attempt solving.
  double rcond_tol = std::numeric_limits<double>::epsilon();

  radial::solve_least_squares_patch(patch_cells, patch_dofs, patch_basis_funcs,
                                    interpolant, basis_size,
                                    rcond_tol, fe_values_nodes, local_dof_indices,
                                    a, rcond);

  // Test equivalence at a point
  Point<dim> test_point = {0.5, 0.5, 0.5};
  // Since the least-squares problem is solved with normalized coordinates,
  // use the normalized coordinates when passing the point in to the patch
  // basis functions.
  Point<dim> test_point_normalized = {0.0, 0.0, 0.0};

  double exact_val = cubic.value(test_point);
  std::cout << "P" << order <<  " exact value = " << exact_val << std::endl;

  double test_val = 0;
  for (int i = 0; i < basis_size; i++)
    test_val += a(i) * patch_basis_funcs[i](test_point_normalized);
  std::cout << "P" << order <<  " test value = " << test_val << std::endl;

  double relative_error = std::abs(exact_val - test_val) / exact_val;

  if (relative_error < 1e-14)
    std::cout << "P" << order << " Passed" << std::endl;
  else
    std::cout << "P" << order << " Failed" << std::endl;
}

int main()
{
  least_squares_patch_test_P1_3D();
  least_squares_patch_test_P2_3D();
}