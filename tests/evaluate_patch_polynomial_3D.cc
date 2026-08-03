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
// Test of evaluating a least-squares patch polynomial in 3D
//-------------------------------------------------------------------------//

using namespace dealii;

void evaluate_patch_polynomial_test_P1_3D()
{
  const int dim = 3;
  const int order = 1;

  //-------------------------------------------------------------------------//
  // Function to get truth value from
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

  // Make mesh extents -1 to 1 instead of 0 to 1 so that patch coordinate
  // normalization does nothing.
  GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, 2, -1, 1);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Base finite element field
  const FE_SimplexP<dim> fe(order);
  MappingP1<dim> mapping;

  DoFHandler<dim> dof_handler(triangulation); 
  dof_handler.distribute_dofs(fe);

  // Nodal coordinates
  std::vector<Point<dim>> dof_coords(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, dof_coords);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Enriched finite element field
  const FE_SimplexP<dim> fe_enriched(order_enriched);

  DoFHandler<dim> dof_handler_enriched(triangulation); 
  dof_handler_enriched.distribute_dofs(fe_enriched);

  // Nodal coordinates
  std::vector<Point<dim>> dof_coords_enriched(dof_handler_enriched.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dof_handler_enriched,
                                       dof_coords_enriched);

  // Enriched solution vector
  Vector<double> solution_enriched(dof_handler_enriched.n_dofs());
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

  // Evaluate least-squares polynomial for central vertex patch
  int v = 13;

  // The extents of the bounding box of the patch for the central vertex
  // coincide with the extents of the domain.
  Point<dim> coord_min = {-1.0, -1.0, -1.0};
  Point<dim> coord_max = { 1.0,  1.0,  1.0};

  // We will pretend that the true polynomial coefficients are the coefficients
  // obtained from a least-squares problem on the patch. The reason we can do
  // this is that we chose the mesh to have extents -1 to 1, which means patch
  // coordinates normalization does nothing. Therefore, a least-squares solve
  // would simply give the true coefficients.
  Vector<double> lsq_coeffs(coeffs.begin(), coeffs.end());

  radial::evaluate_patch_polynomial(dof_coords_enriched,
                                    vertex_to_dof_enriched[v], vertex_to_weight[v],
                                    coord_min, coord_max, patch_basis_funcs,
                                    lsq_coeffs, solution_enriched);

  // Test equivalence at a point
  Point<dim> test_point = { -0.5, 0.0, -0.5};

  double exact_val = quadratic.value(test_point);
  std::cout << "P" << order <<  " exact value = " << exact_val << std::endl;

  // The enriched node at ( -0.5, 0, -0.5) has index 14
  types::global_dof_index test_point_dof_index = 14;

  // Need to add half of the true value because this patch only contributes
  // half due to barycentric weighting.
  double test_val = solution_enriched(test_point_dof_index) + 0.5 * exact_val;
  std::cout << "P" << order <<  " test value = " << test_val << std::endl;

  double relative_error = std::abs(exact_val - test_val) / exact_val;

  if (relative_error < 1e-14)
    std::cout << "P" << order << " Passed (Test Point 1)" << std::endl;
  else
    std::cout << "P" << order << " Failed (Test Point 1)" << std::endl;

  // Test equivalence at another point
  test_point = { 0.0, -0.5, -0.5};

  exact_val = quadratic.value(test_point);
  std::cout << "P" << order <<  " exact value = " << exact_val << std::endl;

  // The enriched node at ( 0, -0.5, -0.5) has index 15
  test_point_dof_index = 15;

  // Need to add half of the true value because this patch only contributes
  // half due to barycentric weighting.
  test_val = solution_enriched(test_point_dof_index) + 0.5 * exact_val;
  std::cout << "P" << order <<  " test value = " << test_val << std::endl;

  relative_error = std::abs(exact_val - test_val) / exact_val;

  if (relative_error < 1e-14)
    std::cout << "P" << order << " Passed (Test Point 2)" << std::endl;
  else
    std::cout << "P" << order << " Failed (Test Point 2)" << std::endl;
}

void evaluate_patch_polynomial_test_P2_3D()
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

  // Make mesh extents -1 to 1 instead of 0 to 1 so that patch coordinate
  // normalization does nothing.
  GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, 2, -1, 1);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Base finite element field
  const FE_SimplexP<dim> fe(order);
  MappingP1<dim> mapping;

  DoFHandler<dim> dof_handler(triangulation); 
  dof_handler.distribute_dofs(fe);

  // Nodal coordinates
  std::vector<Point<dim>> dof_coords(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, dof_coords);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Enriched finite element field
  const FE_SimplexP<dim> fe_enriched(order_enriched);

  DoFHandler<dim> dof_handler_enriched(triangulation); 
  dof_handler_enriched.distribute_dofs(fe_enriched);

  // Nodal coordinates
  std::vector<Point<dim>> dof_coords_enriched(dof_handler_enriched.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dof_handler_enriched,
                                       dof_coords_enriched);

  // Enriched solution vector
  Vector<double> solution_enriched(dof_handler_enriched.n_dofs());
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

  // Evaluate least-squares polynomial for central vertex patch
  int v = 4;

  // The extents of the bounding box of the patch for the central vertex
  // coincide with the extents of the domain.
  Point<dim> coord_min = {-1.0, -1.0, -1.0};
  Point<dim> coord_max = { 1.0,  1.0,  1.0};

  // We will pretend that the true polynomial coefficients are the coefficients
  // obtained from a least-squares problem on the patch. The reason we can do
  // this is that we chose the mesh to have extents -1 to 1, which means patch
  // coordinates normalization does nothing. Therefore, a least-squares solve
  // would simply give the true coefficients.
  Vector<double> lsq_coeffs(coeffs.begin(), coeffs.end());

  radial::evaluate_patch_polynomial(dof_coords_enriched,
                                    vertex_to_dof_enriched[v], vertex_to_weight[v],
                                    coord_min, coord_max, patch_basis_funcs,
                                    lsq_coeffs, solution_enriched);

  // Test equivalence at a point
  Point<dim> test_point = { 0.0, -2.0/3.0, -1.0};

  double exact_val = cubic.value(test_point);
  std::cout << "P" << order <<  " exact value = " << exact_val << std::endl;

  // The enriched node at ( 0, -2/3, -1) has index 22
  types::global_dof_index test_point_dof_index = 22;

  // Need to add 2/3 of the true value because this patch only contributes
  // 1/3 due to barycentric weighting.
  double test_val = solution_enriched(test_point_dof_index) + (2.0/3.0) * exact_val;
  std::cout << "P" << order <<  " test value = " << test_val << std::endl;

  double relative_error = std::abs(exact_val - test_val) / exact_val;

  if (relative_error < 1e-14)
    std::cout << "P" << order << " Passed (Test Point 1)" << std::endl;
  else
    std::cout << "P" << order << " Failed (Test Point 1)" << std::endl;

  // Test equivalence at another point
  test_point = { -2.0/3.0, 0.0, -1.0};

  exact_val = cubic.value(test_point);
  std::cout << "P" << order <<  " exact value = " << exact_val << std::endl;

  // The enriched node at ( -2/3, 0, -1) has index 25
  test_point_dof_index = 25;

  // Need to add 2/3 of the true value because this patch only contributes
  // 1/3 due to barycentric weighting.
  test_val = solution_enriched(test_point_dof_index) + (2.0/3.0) * exact_val;
  std::cout << "P" << order <<  " test value = " << test_val << std::endl;

  relative_error = std::abs(exact_val - test_val) / exact_val;

  if (relative_error < 1e-14)
    std::cout << "P" << order << " Passed (Test Point 2)" << std::endl;
  else
    std::cout << "P" << order << " Failed (Test Point 2)" << std::endl;
}

int main()
{
  evaluate_patch_polynomial_test_P1_3D();
  evaluate_patch_polynomial_test_P2_3D();
}