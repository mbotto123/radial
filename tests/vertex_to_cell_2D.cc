#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include <recovery_common.h>

//-------------------------------------------------------------------------//
// Test of creating vertex-to-cell mapping in 2D
//-------------------------------------------------------------------------//

using namespace dealii;

void vertex_to_cell_test_2D(const int order)
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

  for (types::global_vertex_index v = 0; v < vertex_to_cell.size(); v++)
  {
    std::cout << "Patch " << v << " cells: ";
    for (const auto &cell: vertex_to_cell[v])
    {
      std::cout << cell->index() << ",";
    }
    std::cout << std::endl;
  }

  for (types::global_vertex_index v = 0; v < vertex_to_cell_enriched.size(); v++)
  {
    std::cout << "Enriched Patch " << v << " cells: ";
    for (const auto &cell: vertex_to_cell_enriched[v])
    {
      std::cout << cell->index() << ",";
    }
    std::cout << std::endl;
  }
}

int main()
{
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "P1: " << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  vertex_to_cell_test_2D(1);
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "P2: " << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  vertex_to_cell_test_2D(2);
  std::cout << "----------------------------------------" << std::endl;
}