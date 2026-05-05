#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include <solution_recovery.h>

//-------------------------------------------------------------------------//  
// Test of solution recovery function
//-------------------------------------------------------------------------//  

using namespace dealii;

// Function to interpolate from
template <int dim>
class Solution : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template<int dim>
double Solution<dim>::value(const Point<dim> &p,
                            const unsigned int /*component*/) const
{
  if (dim == 2)
  {
    double x = p[0];
    double y = p[1];
    return 1 - 3*x + 4*y + 2*x*x - 5*y*y + 7*x*y;
  }
  else if (dim == 3)
  {
    double x = p[0];
    double y = p[1];
    double z = p[2];
    return 1 - 3*x + 4*y + z + 2*x*x - 5*y*y + 7*x*y + 6*x*z - 9*y*z + z*z;
  }
}

void ppr_interpolant_test_2D()
{
  const int dim = 2;

  //-------------------------------------------------------------------------//  
  // Mesh
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, 2);
  //-------------------------------------------------------------------------//  

  //-------------------------------------------------------------------------//  
  // Base finite element field
  const int order = 1;
  const FE_SimplexP<dim> fe(order);
  MappingP1<dim> mapping;

  DoFHandler<dim> dof_handler(triangulation); 
  dof_handler.distribute_dofs(fe);
  //-------------------------------------------------------------------------//  

  //-------------------------------------------------------------------------//  
  // Compute linear interpolant
  Vector<double> interpolant(dof_handler.n_dofs());

  VectorTools::interpolate(dof_handler, Solution<dim>(), interpolant);
  //-------------------------------------------------------------------------//  

  //-------------------------------------------------------------------------//  
  // Enriched finite element field
  const int order_enriched = order + 1;
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
                                    Solution<dim>(),
                                    cell_errors,
                                    error_quadrature,
                                    norm);
  double global_error = VectorTools::compute_global_error(triangulation,
                                                          cell_errors,
                                                          norm);

  if (global_error < 1e-14)
    std::cout << "2D Passed" << std::endl;
  else
    std::cout << "2D Failed" << std::endl;
  //-------------------------------------------------------------------------//
}

void ppr_interpolant_test_3D()
{
  const int dim = 3;

  //-------------------------------------------------------------------------//  
  // Mesh
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, 2);
  //-------------------------------------------------------------------------//  

  //-------------------------------------------------------------------------//  
  // Base finite element field
  const int order = 1;
  const FE_SimplexP<dim> fe(order);
  MappingP1<dim> mapping;

  DoFHandler<dim> dof_handler(triangulation); 
  dof_handler.distribute_dofs(fe);
  //-------------------------------------------------------------------------//  

  //-------------------------------------------------------------------------//  
  // Compute linear interpolant
  Vector<double> interpolant(dof_handler.n_dofs());

  VectorTools::interpolate(dof_handler, Solution<dim>(), interpolant);
  //-------------------------------------------------------------------------//  

  //-------------------------------------------------------------------------//  
  // Enriched finite element field
  const int order_enriched = order + 1;
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
                                    Solution<dim>(),
                                    cell_errors,
                                    error_quadrature,
                                    norm);
  double global_error = VectorTools::compute_global_error(triangulation,
                                                          cell_errors,
                                                          norm);

  if (global_error < 1e-14)
    std::cout << "3D Passed" << std::endl;
  else
    std::cout << "3D Failed" << std::endl;
  //-------------------------------------------------------------------------//
}

int main()
{
  ppr_interpolant_test_2D();
  ppr_interpolant_test_3D();
}
