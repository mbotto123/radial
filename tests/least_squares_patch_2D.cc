#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

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
// Test of solving a least-squares problem on a patch in 2D
//-------------------------------------------------------------------------//

using namespace dealii;

void least_squares_patch_discrete_test_P1_2D()
{
  const int dim = 2;
  const int order = 1;

  //-------------------------------------------------------------------------//
  // Function to interpolate from
  const int order_enriched = order + 1;

  const int basis_size = radial::get_min_points<dim>(order_enriched);

  // Monomial coefficients
  std::vector<double> coeffs(basis_size);
  coeffs = {1, -3, 4, 2, 7, -5};

  // Monomial exponents
  const double exponents_array[] = {0, 0,  // x^0 y^0
                                    1, 0,  // x^1 y^0
                                    0, 1,  // x^0 y^1
                                    2, 0,  // x^2 y^0
                                    1, 1,  // x^1 y^1
                                    0, 2}; // x^0 y^2
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

  // Nodal coordinates
  std::vector<Point<dim>> dof_coords(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, dof_coords);
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

  radial::patch_basis<dim> patch_basis_funcs(basis_size);
  radial::create_patch_basis(order, patch_basis_funcs);
  //-------------------------------------------------------------------------//

  std::vector<std::set<types::global_vertex_index>> vertex_to_vertex;
  std::vector<std::set<types::global_dof_index>> vertex_to_dof;
  std::vector<std::vector<types::global_dof_index>> vertex_to_dof_enriched;
  std::vector<std::vector<double>> vertex_to_weight;

  radial::create_vertex_mappings(dof_handler, dof_handler_enriched,
                                 vertex_to_vertex, vertex_to_dof,
                                 vertex_to_dof_enriched, vertex_to_weight);

  // Solve least-squares problem for central vertex patch. This patch does not
  // require any growth iterations.
  int v = 4;
  std::set<types::global_dof_index> patch_dofs = vertex_to_dof[v];

  // Vector of least-squares coefficients
  Vector<double> a(basis_size);
  // Reciprocal condition number of the least-squares system on the patch
  double rcond;

  Point<dim> coord_min, coord_max;
  radial::find_patch_bounding_box(dof_coords, patch_dofs,
                                  coord_min, coord_max);
  radial::least_squares_patch_discrete(dof_coords, interpolant,
                                       patch_dofs, basis_size,
                                       coord_min, coord_max, patch_basis_funcs,
                                       a, rcond);

  // Test equivalence at a point
  Point<dim> test_point = {0.5, 0.5};
  // Since the least-squares problem is solved with normalized coordinates,
  // use the normalized coordinates when passing the point in to the patch
  // basis functions.
  Point<dim> test_point_normalized = {0.0, 0.0};

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

void least_squares_patch_test_P1_2D()
{
  const int dim = 2;
  const int order = 1;

  //-------------------------------------------------------------------------//
  // Function to interpolate from
  const int order_enriched = order + 1;

  const int basis_size = radial::get_min_points<dim>(order_enriched);

  // Monomial coefficients
  std::vector<double> coeffs(basis_size);
  coeffs = {1, -3, 4, 2, 7, -5};

  // Monomial exponents
  const double exponents_array[] = {0, 0,  // x^0 y^0
                                    1, 0,  // x^1 y^0
                                    0, 1,  // x^0 y^1
                                    2, 0,  // x^2 y^0
                                    1, 1,  // x^1 y^1
                                    0, 2}; // x^0 y^2
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

  radial::patch_basis<dim> patch_basis_funcs(basis_size);
  radial::create_patch_basis(order, patch_basis_funcs);
  //-------------------------------------------------------------------------//

  std::vector<std::vector<radial::cell_pointer<dim>>> vertex_to_cell;
  std::vector<std::vector<radial::cell_pointer<dim>>> vertex_to_cell_enriched;

  radial::create_vertex_to_cell(dof_handler, dof_handler_enriched,
                                vertex_to_cell, vertex_to_cell_enriched);

  std::set<types::global_dof_index> patch_dofs;

  // Solve least-squares problem for central vertex patch. This patch does not
  // require any growth iterations.
  int v = 4;
  std::vector<radial::cell_pointer<dim>> patch_cells = vertex_to_cell[v];
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

  Point<dim> coord_min, coord_max;
  radial::find_patch_bounding_box(patch_cells, patch_dofs,
                                  fe_values_nodes, local_dof_indices,
                                  coord_min, coord_max);
  radial::least_squares_patch(patch_cells, patch_dofs,
                              coord_min, coord_max,
                              patch_basis_funcs, fe,
                              interpolant, fe_values_nodes,
                              a, rcond);

  // Test equivalence at a point
  Point<dim> test_point = {0.5, 0.5};
  // Since the least-squares problem is solved with normalized coordinates,
  // use the normalized coordinates when passing the point in to the patch
  // basis functions.
  Point<dim> test_point_normalized = {0.0, 0.0};

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

void least_squares_patch_discrete_test_P2_2D()
{
  const int dim = 2;
  const int order = 2;

  //-------------------------------------------------------------------------//
  // Function to interpolate from
  const int order_enriched = order + 1;

  const int basis_size = radial::get_min_points<dim>(order_enriched);

  // Monomial coefficients
  std::vector<double> coeffs(basis_size);
  coeffs = {1, -3, 4, 2, 7, -5, 6, 10, -7, -1};

  // Monomial exponents
  const double exponents_array[] = {0, 0,  // x^0 y^0
                                    1, 0,  // x^1 y^0
                                    0, 1,  // x^0 y^1
                                    2, 0,  // x^2 y^0
                                    1, 1,  // x^1 y^1
                                    0, 2,  // x^0 y^2
                                    3, 0,  // x^3 y^0
                                    2, 1,  // x^2 y^1
                                    1, 2,  // x^1 y^2
                                    0, 3}; // x^0 y^3
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

  // Nodal coordinates
  std::vector<Point<dim>> dof_coords(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, dof_coords);
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

  radial::patch_basis<dim> patch_basis_funcs(basis_size);
  radial::create_patch_basis(order, patch_basis_funcs);
  //-------------------------------------------------------------------------//

  std::vector<std::set<types::global_vertex_index>> vertex_to_vertex;
  std::vector<std::set<types::global_dof_index>> vertex_to_dof;
  std::vector<std::vector<types::global_dof_index>> vertex_to_dof_enriched;
  std::vector<std::vector<double>> vertex_to_weight;

  radial::create_vertex_mappings(dof_handler, dof_handler_enriched,
                                 vertex_to_vertex, vertex_to_dof,
                                 vertex_to_dof_enriched, vertex_to_weight);

  // Solve least-squares problem for central vertex patch. This patch does not
  // require any growth iterations.
  int v = 4;
  std::set<types::global_dof_index> patch_dofs = vertex_to_dof[v];

  // Vector of least-squares coefficients
  Vector<double> a(basis_size);
  // Reciprocal condition number of the least-squares system on the patch
  double rcond;

  Point<dim> coord_min, coord_max;
  radial::find_patch_bounding_box(dof_coords, patch_dofs,
                                  coord_min, coord_max);
  radial::least_squares_patch_discrete(dof_coords, interpolant,
                                       patch_dofs, basis_size,
                                       coord_min, coord_max, patch_basis_funcs,
                                       a, rcond);

  // Test equivalence at a point
  Point<dim> test_point = {0.5, 0.5};
  // Since the least-squares problem is solved with normalized coordinates,
  // use the normalized coordinates when passing the point in to the patch
  // basis functions.
  Point<dim> test_point_normalized = {0.0, 0.0};

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

void least_squares_patch_test_P2_2D()
{
  const int dim = 2;
  const int order = 2;

  //-------------------------------------------------------------------------//
  // Function to interpolate from
  const int order_enriched = order + 1;

  const int basis_size = radial::get_min_points<dim>(order_enriched);

  // Monomial coefficients
  std::vector<double> coeffs(basis_size);
  coeffs = {1, -3, 4, 2, 7, -5, 6, 10, -7, -1};

  // Monomial exponents
  const double exponents_array[] = {0, 0,  // x^0 y^0
                                    1, 0,  // x^1 y^0
                                    0, 1,  // x^0 y^1
                                    2, 0,  // x^2 y^0
                                    1, 1,  // x^1 y^1
                                    0, 2,  // x^0 y^2
                                    3, 0,  // x^3 y^0
                                    2, 1,  // x^2 y^1
                                    1, 2,  // x^1 y^2
                                    0, 3}; // x^0 y^3
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

  radial::patch_basis<dim> patch_basis_funcs(basis_size);
  radial::create_patch_basis(order, patch_basis_funcs);
  //-------------------------------------------------------------------------//

  std::vector<std::vector<radial::cell_pointer<dim>>> vertex_to_cell;
  std::vector<std::vector<radial::cell_pointer<dim>>> vertex_to_cell_enriched;

  radial::create_vertex_to_cell(dof_handler, dof_handler_enriched,
                                vertex_to_cell, vertex_to_cell_enriched);

  std::set<types::global_dof_index> patch_dofs;

  // Solve least-squares problem for central vertex patch. This patch does not
  // require any growth iterations.
  int v = 4;
  std::vector<radial::cell_pointer<dim>> patch_cells = vertex_to_cell[v];
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

  Point<dim> coord_min, coord_max;
  radial::find_patch_bounding_box(patch_cells, patch_dofs,
                                  fe_values_nodes, local_dof_indices,
                                  coord_min, coord_max);
  radial::least_squares_patch(patch_cells, patch_dofs,
                              coord_min, coord_max,
                              patch_basis_funcs, fe,
                              interpolant, fe_values_nodes,
                              a, rcond);

  // Test equivalence at a point
  Point<dim> test_point = {0.5, 0.5};
  // Since the least-squares problem is solved with normalized coordinates,
  // use the normalized coordinates when passing the point in to the patch
  // basis functions.
  Point<dim> test_point_normalized = {0.0, 0.0};

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
  least_squares_patch_discrete_test_P1_2D();
  least_squares_patch_discrete_test_P2_2D();

  least_squares_patch_test_P1_2D();
  least_squares_patch_test_P2_2D();
}