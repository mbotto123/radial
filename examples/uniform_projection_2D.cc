#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/base/convergence_table.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include <solution_recovery.h>

using namespace dealii;

//-------------------------------------------------------------------------//
// L^2 Projection with uniform refinement. Solution recovery is performed
// on the projected solution.
//-------------------------------------------------------------------------//

// Function to project from
// Double boundary layer in 2D / Triple boundary layer in 3D
//
// Reference for this function:
// Galbraith et al. "Verification of Unstructured Grid Adaptation Components".
// AIAA Journal 58.9 (2020).
template <int dim>
class Solution : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &p,
                                  const unsigned int component = 0) const override;
};

template<int dim>
double Solution<dim>::value(const Point<dim> &p,
                            const unsigned int /*component*/) const
{
  // Advection
  Tensor<1, dim> a;
  for (int d = 0; d < dim; d++)
    a[d] = 1.0;

  // Diffusion
  double nu = 1./30.;

  double mult_term = 1.0;
  for (int d = 0; d < dim; d++)
    mult_term *= (1 - std::exp(-a[d]*(1 - p[d])/nu)) / (1 - std::exp(-a[d]/nu));

  return -mult_term + 1;
}

template<int dim>
Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p,
                                       const unsigned int /*component*/) const
{
  // Advection
  Tensor<1, dim> a;
  for (int d = 0; d < dim; d++)
    a[d] = 1.0;

  // Diffusion
  double nu = 1./30.;

  Tensor<1, dim> num;
  double denom = nu;
  for (int d = 0; d < dim; d++)
  {
    denom *= std::exp(a[d]/nu) - 1;

    num[d] = a[d] * std::exp(a[d]*p[d]/nu);

    for (int dd = 0; dd < dim; dd++)
    {
      // Generally not advisable to have an if statement in a function
      // that gets called a lot. Is there a way to rewrite this without
      // using an if statement?
      if (dd != d)
        num[d] *= std::exp(a[dd]/nu) - std::exp(a[dd]*p[dd]/nu);
    }
  }

  return num / denom;
}

int main()
{
  //-------------------------------------------------------------------------//
  // Run parameters
  const int dim = 2;

  const int order = 2;

  int max_level = 7;
  //-------------------------------------------------------------------------//

  ConvergenceTable convergence_table;
  
  for (int level = 1; level < max_level; level++)
  {
    //-------------------------------------------------------------------------//
    // Mesh
    int repetitions = std::pow(2, level);

    Triangulation<dim> triangulation;
    GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, repetitions);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Base finite element field
    const FE_SimplexP<dim> fe(order);
    MappingP1<dim> mapping;

    DoFHandler<dim> dof_handler(triangulation); 
    dof_handler.distribute_dofs(fe);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Compute L2 projection
    Vector<double> solution(dof_handler.n_dofs());

    // Empty constraints object (no constraints to enforce)
    AffineConstraints<double> cm;
    cm.close();

    VectorTools::project(dof_handler, cm, QGaussSimplex<dim>(order + 2),
                         Solution<dim>(), solution);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Enriched finite element field
    const int order_enriched = order + 1;
    const FE_SimplexP<dim> fe_enriched(order_enriched);

    DoFHandler<dim> dof_handler_enriched(triangulation); 
    dof_handler_enriched.distribute_dofs(fe_enriched);

    Vector<double> solution_enriched(dof_handler_enriched.n_dofs());
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Solution recovery
    radial::recover_solution_ppr(dof_handler, mapping, solution,
                                 dof_handler_enriched, solution_enriched);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Compute L2 projection from enriched solution
    Vector<double> projection_from_enriched(dof_handler.n_dofs());

    Functions::FEFieldFunction<dim> fe_function_enriched(dof_handler_enriched,
                                                         solution_enriched,
                                                         mapping);

    VectorTools::project(dof_handler, cm, QGaussSimplex<dim>(order + 2),
                         fe_function_enriched, projection_from_enriched);
    //-------------------------------------------------------------------------//
    
    //-------------------------------------------------------------------------//
    // Compute error between enriched solution and exact solution
    // (exact solution should be captured exactly by enriched solution)
    QWitherdenVincentSimplex<dim> error_quadrature(order_enriched + 2);

    Vector<double> cell_errors(triangulation.n_cells());

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      cell_errors,
                                      error_quadrature,
                                      VectorTools::L2_norm);
    double L2_error = VectorTools::compute_global_error(triangulation,
                                                        cell_errors,
                                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(mapping,
                                      dof_handler_enriched,
                                      solution_enriched,
                                      Solution<dim>(),
                                      cell_errors,
                                      error_quadrature,
                                      VectorTools::L2_norm);
    double L2_error_enriched = VectorTools::compute_global_error(triangulation,
                                                                 cell_errors,
                                                                 VectorTools::L2_norm);

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      cell_errors,
                                      error_quadrature,
                                      VectorTools::H1_seminorm);
    double H1_error = VectorTools::compute_global_error(triangulation,
                                                        cell_errors,
                                                        VectorTools::H1_seminorm);

    VectorTools::integrate_difference(mapping,
                                      dof_handler_enriched,
                                      solution_enriched,
                                      Solution<dim>(),
                                      cell_errors,
                                      error_quadrature,
                                      VectorTools::H1_seminorm);
    double H1_error_enriched = VectorTools::compute_global_error(triangulation,
                                                                 cell_errors,
                                                                 VectorTools::H1_seminorm);

    // Compute L2 difference between enriched solution and projection from it
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      projection_from_enriched,
                                      fe_function_enriched,
                                      cell_errors,
                                      QGaussSimplex<dim>(order + 2),
                                      VectorTools::L2_norm);
    double L2_diff = VectorTools::compute_global_error(triangulation,
                                                       cell_errors,
                                                       VectorTools::L2_norm);
    double theta_L2 = L2_diff / L2_error;

    convergence_table.add_value("level", level);
    convergence_table.add_value("cells", triangulation.n_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("L2E", L2_error_enriched);
    convergence_table.add_value("H1", H1_error);
    convergence_table.add_value("H1E", H1_error_enriched);
    convergence_table.add_value("theta_L2", theta_L2);
    //-------------------------------------------------------------------------//
  }

  convergence_table.set_precision("L2", 3);
  convergence_table.set_precision("L2E", 3);
  convergence_table.set_precision("H1", 3);
  convergence_table.set_precision("H1E", 3);
  convergence_table.set_precision("theta_L2", 3);
  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("L2E", true);
  convergence_table.set_scientific("H1", true);
  convergence_table.set_scientific("H1E", true);

  std::cout << std::endl;
  convergence_table.write_text(std::cout);

  convergence_table.add_column_to_supercolumn("level", "n cells");
  convergence_table.add_column_to_supercolumn("cells", "n cells");

  std::vector<std::string> new_order;
  new_order.emplace_back("n cells");
  new_order.emplace_back("H1");
  new_order.emplace_back("H1E");
  new_order.emplace_back("L2");
  new_order.emplace_back("L2E");
  convergence_table.set_column_order(new_order);

  convergence_table.evaluate_convergence_rates(
    "L2", ConvergenceTable::reduction_rate_log2);
  convergence_table.evaluate_convergence_rates(
    "L2E", ConvergenceTable::reduction_rate_log2);
  convergence_table.evaluate_convergence_rates(
    "H1", ConvergenceTable::reduction_rate_log2);
  convergence_table.evaluate_convergence_rates(
    "H1E", ConvergenceTable::reduction_rate_log2);

  std::cout << std::endl;
  convergence_table.write_text(std::cout);
}
