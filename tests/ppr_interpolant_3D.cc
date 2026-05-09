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
// Test of solution recovery function in 3D
//-------------------------------------------------------------------------//

using namespace dealii;

// Functions to interpolate from

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

void ppr_P1_interpolant_test_3D()
{
  const int dim = 3;
  const int order = 1;

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

  VectorTools::interpolate(dof_handler, Quadratic<dim>(), interpolant);
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
                                    Quadratic<dim>(),
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

  VectorTools::interpolate(dof_handler, Quadratic<dim>(), interpolant);
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
                                    Quadratic<dim>(),
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
