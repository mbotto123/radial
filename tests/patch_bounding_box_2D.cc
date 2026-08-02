#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include <recovery_common.h>

//-------------------------------------------------------------------------//
// Test of finding a patch bounding box in 2D
//-------------------------------------------------------------------------//

using namespace dealii;

void patch_bounding_box_test_2D_linear_mesh(const int order)
{
  const int dim = 2;

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
  // Enriched finite element field
  const int order_enriched = order + 1;
  const FE_SimplexP<dim> fe_enriched(order_enriched);

  DoFHandler<dim> dof_handler_enriched(triangulation); 
  dof_handler_enriched.distribute_dofs(fe_enriched);
  //-------------------------------------------------------------------------//

  std::vector<std::vector<radial::cell_pointer<dim>>> vertex_to_cell;
  std::vector<std::vector<radial::cell_pointer<dim>>> vertex_to_cell_enriched;

  radial::create_vertex_to_cell(dof_handler, dof_handler_enriched,
                                vertex_to_cell, vertex_to_cell_enriched);

  std::set<types::global_vertex_index> patch_vertices;

  // Test bounding box for central vertex patch. Should coincide with the
  // extents of the domain, i.e. 0 to 1 in all directions.
  int v = 4;
  std::vector<radial::cell_pointer<dim>> patch_cells = vertex_to_cell[v];
  for (const auto &cell: vertex_to_cell[v])
  {
    for (const auto v_patch: cell->vertex_indices())
      patch_vertices.insert(cell->vertex_index(v_patch));
  }

  Point<dim> coord_min, coord_max;
  radial::find_patch_bounding_box(dof_handler, patch_vertices,
                                  coord_min, coord_max);

  std::cout << "Min x = " << coord_min(0) << std::endl;
  std::cout << "Min y = " << coord_min(1) << std::endl;
  std::cout << "Max x = " << coord_max(0) << std::endl;
  std::cout << "Max y = " << coord_max(1) << std::endl;
}

void patch_bounding_box_test_2D_general_mesh_nodal(const int order)
{
  const int dim = 2;

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

  // Nodal coordinates
  std::vector<Point<dim>> dof_coords(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, dof_coords);
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Enriched finite element field
  const int order_enriched = order + 1;
  const FE_SimplexP<dim> fe_enriched(order_enriched);

  DoFHandler<dim> dof_handler_enriched(triangulation); 
  dof_handler_enriched.distribute_dofs(fe_enriched);
  //-------------------------------------------------------------------------//

  std::vector<std::set<types::global_vertex_index>> vertex_to_vertex;
  std::vector<std::set<types::global_dof_index>> vertex_to_dof;
  std::vector<std::vector<types::global_dof_index>> vertex_to_dof_enriched;
  std::vector<std::vector<double>> vertex_to_weight;

  radial::create_vertex_mappings(dof_handler, dof_handler_enriched,
                                 vertex_to_vertex, vertex_to_dof,
                                 vertex_to_dof_enriched, vertex_to_weight);

  // Test bounding box for central vertex patch. Should coincide with the
  // extents of the domain, i.e. 0 to 1 in all directions.
  int v = 4;
  std::set<types::global_dof_index> patch_dofs = vertex_to_dof[v];

  Point<dim> coord_min, coord_max;
  radial::find_patch_bounding_box(dof_coords, patch_dofs,
                                  coord_min, coord_max);

  std::cout << "Min x = " << coord_min(0) << std::endl;
  std::cout << "Min y = " << coord_min(1) << std::endl;
  std::cout << "Max x = " << coord_max(0) << std::endl;
  std::cout << "Max y = " << coord_max(1) << std::endl;
}

void patch_bounding_box_test_2D_general_mesh_elemental(const int order)
{
  const int dim = 2;

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
  // Enriched finite element field
  const int order_enriched = order + 1;
  const FE_SimplexP<dim> fe_enriched(order_enriched);

  DoFHandler<dim> dof_handler_enriched(triangulation); 
  dof_handler_enriched.distribute_dofs(fe_enriched);
  //-------------------------------------------------------------------------//

  std::vector<std::vector<radial::cell_pointer<dim>>> vertex_to_cell;
  std::vector<std::vector<radial::cell_pointer<dim>>> vertex_to_cell_enriched;

  radial::create_vertex_to_cell(dof_handler, dof_handler_enriched,
                                vertex_to_cell, vertex_to_cell_enriched);

  std::set<types::global_dof_index> patch_dofs;

  // Test bounding box for central vertex patch. Should coincide with the
  // extents of the domain, i.e. 0 to 1 in all directions.
  int v = 4;
  std::vector<radial::cell_pointer<dim>> patch_cells = vertex_to_cell[v];
  for (const auto &cell: vertex_to_cell[v])
  {
    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i : fe_values_nodes.dof_indices())
      patch_dofs.insert(local_dof_indices[i]);
  }

  Point<dim> coord_min, coord_max;
  radial::find_patch_bounding_box(patch_cells, patch_dofs,
                                  fe_values_nodes, local_dof_indices,
                                  coord_min, coord_max);

  std::cout << "Min x = " << coord_min(0) << std::endl;
  std::cout << "Min y = " << coord_min(1) << std::endl;
  std::cout << "Max x = " << coord_max(0) << std::endl;
  std::cout << "Max y = " << coord_max(1) << std::endl;
}

int main()
{
  patch_bounding_box_test_2D_linear_mesh(1);
  patch_bounding_box_test_2D_linear_mesh(2);

  // Still testing on linear meshes, but the bounding box function does not
  // assume they are linear in its implementation.
  patch_bounding_box_test_2D_general_mesh_nodal(1);
  patch_bounding_box_test_2D_general_mesh_nodal(2);

  patch_bounding_box_test_2D_general_mesh_elemental(1);
  patch_bounding_box_test_2D_general_mesh_elemental(2);
}