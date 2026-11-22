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
// Test of creating vertex mappings in 2D
//-------------------------------------------------------------------------//

using namespace dealii;

void vertex_mappings_test_2D(const int order)
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

  std::vector<std::set<types::global_vertex_index>> vertex_to_vertex;
  std::vector<std::set<types::global_dof_index>> vertex_to_dof;
  std::vector<std::vector<types::global_dof_index>> vertex_to_dof_enriched;
  std::vector<std::vector<double>> vertex_to_weight;

  radial::create_vertex_mappings(dof_handler, dof_handler_enriched,
                                 vertex_to_vertex, vertex_to_dof,
                                 vertex_to_dof_enriched, vertex_to_weight);

  for (types::global_vertex_index v = 0; v < vertex_to_vertex.size(); v++)
  {
    std::cout << "Patch " << v << " vertices: ";
    for (const auto& v_patch : vertex_to_vertex[v])
    {
      std::cout << v_patch << ",";
    }
    std::cout << std::endl;

    std::cout << "Patch " << v << " DOFs: ";
    for (const auto& dof_patch : vertex_to_dof[v])
    {
      std::cout << dof_patch << ",";
    }
    std::cout << std::endl << std::endl;
  }

  for (types::global_vertex_index v = 0; v < vertex_to_vertex.size(); v++)
  {
    std::cout << "Enriched Patch " << v << " DOFs: ";
    for (const auto& dof_patch : vertex_to_dof_enriched[v])
    {
      std::cout << dof_patch << ",";
    }
    std::cout << std::endl;

    std::cout << "Enriched Patch " << v << " weights: ";
    for (const auto& weight_patch : vertex_to_weight[v])
    {
      std::cout << weight_patch << ",";
    }
    std::cout << std::endl << std::endl;
  }
}

int main()
{
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "P1: " << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  vertex_mappings_test_2D(1);
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "P2: " << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  vertex_mappings_test_2D(2);
  std::cout << "----------------------------------------" << std::endl;
}