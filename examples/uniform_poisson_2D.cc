#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/convergence_table.h>

#include <deal.II/base/timer.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include <solution_recovery.h>

using namespace dealii;

//-------------------------------------------------------------------------//
// Poisson equation with uniform refinement. Solution recovery is performed
// on the solution.
//-------------------------------------------------------------------------//

template <int dim>
class ExactSolution : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &p,
                                  const unsigned int component = 0) const override;
};

template<int dim>
double ExactSolution<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
  double value = 1;
  for (int d = 0; d < dim; d++)
    value *= std::sin(2 * numbers::PI * p[d]);

  return value;
}

template<int dim>
Tensor<1, dim> ExactSolution<dim>::gradient(const Point<dim> &p,
                                            const unsigned int /*component*/) const
{
  Tensor<1, dim> gradient;
  for (int d = 0; d < dim; d++)
  {
    gradient[d] = 2 * numbers::PI * std::cos(2 * numbers::PI * p[d]);

    for (int dd = 0; dd < dim; dd++)
    {
      // Generally not advisable to have an if statement in a function
      // that gets called a lot. Is there a way to rewrite this without
      // using an if statement?
      if (dd != d)
        gradient[d] *= std::sin(2 * numbers::PI * p[dd]);
    }
  }

  return gradient;
}

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

// RHS by method of manufactured solutions
template<int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
  double multiplier = 4.0 * dim;

  double value = multiplier * numbers::PI * numbers::PI;
  for (int d = 0; d < dim; d++)
    value *= std::sin(2 * numbers::PI * p[d]);

  return value;
}

int main()
{
  //-------------------------------------------------------------------------//
  // Run parameters

  const int dim = 2;
  const int min_order = 1;
  const int max_order = 2;

  // Number of subdivisions in each direction for the coarsest mesh
  const int min_repetitions = 4;
  // Number of mesh refinements
  const int max_level = 10;

  bool use_iterative_solver = false;
  //-------------------------------------------------------------------------//

  for (int order = min_order; order <= max_order; order++)
  {
    std::cout << std::endl << "Running P" << order << " case..." << std::endl;

    ConvergenceTable convergence_table;
    std::vector<double> h_vals(max_level);
    std::vector<double> L2_vals(max_level), L2E_vals(max_level);
    std::vector<double> H1_vals(max_level), H1E_vals(max_level);
    std::vector<double> L2_rates(max_level), L2E_rates(max_level);
    std::vector<double> H1_rates(max_level), H1E_rates(max_level);

    for (int level = 0; level < max_level; level++)
    {
      //-------------------------------------------------------------------------//
      // Mesh
      Timer timer;

      int repetitions = min_repetitions * std::pow(std::sqrt(2.0), level);

      Triangulation<dim> triangulation;
      GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, repetitions);

      timer.stop();
      std::cout << std::endl << "Level " << level << " timing:" << std::endl;
      std::cout << "  Mesh    : " << timer.last_wall_time() << std::endl;
      //-------------------------------------------------------------------------//

      //-------------------------------------------------------------------------//
      // Base finite element field
      timer.start();

      const FE_SimplexP<dim> fe(order);
      MappingP1<dim> mapping;

      DoFHandler<dim> dof_handler(triangulation); 
      dof_handler.distribute_dofs(fe);

      timer.stop();
      std::cout << "  FE init : " << timer.last_wall_time() << std::endl;
      //-------------------------------------------------------------------------//

      //-------------------------------------------------------------------------//
      // Allocate memory for system matrix, RHS, and solution
      timer.start();

      SparsityPattern sparsity_pattern;
      DynamicSparsityPattern dsp(dof_handler.n_dofs());

      DoFTools::make_sparsity_pattern(dof_handler, dsp);
      sparsity_pattern.copy_from(dsp);

      SparseMatrix<double> system_matrix(sparsity_pattern);
      Vector<double> system_rhs(dof_handler.n_dofs());
      Vector<double> solution(dof_handler.n_dofs());

      timer.stop();
      std::cout << "  Memory  : " << timer.last_wall_time() << std::endl;
      //-------------------------------------------------------------------------//

      //-------------------------------------------------------------------------//
      // Construct Poisson matrix and RHS
      timer.start();

      QGaussSimplex<dim> solution_quadrature(order + 1); // integrate 2p + 1 exactly

      MatrixTools::create_laplace_matrix(mapping, dof_handler,
                                         solution_quadrature, system_matrix,
                                         RightHandSide<dim>(), system_rhs);

      // Apply Dirichlet BCs
      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(dof_handler,
                                               types::boundary_id(0),
                                               ExactSolution<dim>(),
                                               boundary_values);
      MatrixTools::apply_boundary_values(boundary_values,
                                         system_matrix,
                                         solution,
                                         system_rhs);

      timer.stop();
      std::cout << "  Assembly: " << timer.last_wall_time() << std::endl;
      //-------------------------------------------------------------------------//

      //-------------------------------------------------------------------------//
      // Solve system
      timer.start();

      if (use_iterative_solver)
      {
        SolverControl            solver_control(1000, 1e-6 * system_rhs.l2_norm());
        SolverCG<Vector<double>> solver(solver_control);
        solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
      
        std::cout << solver_control.last_step()
                  << " CG iterations needed to obtain convergence." << std::endl;
      }
      else
      {
        SparseDirectUMFPACK solver;

        solution = system_rhs;
        solver.solve(system_matrix, solution);
      }

      timer.stop();
      std::cout << "  Solve   : " << timer.last_wall_time() << std::endl;
      //-------------------------------------------------------------------------//

      //-------------------------------------------------------------------------//
      // Enriched finite element field
      const int order_enriched = order + 1;
      const FE_SimplexP<dim> fe_enriched(order_enriched);

      DoFHandler<dim> dof_handler_enriched(triangulation); 
      dof_handler_enriched.distribute_dofs(fe_enriched);
      //-------------------------------------------------------------------------//

      //-------------------------------------------------------------------------//
      // Solution recovery
      timer.start();

      Vector<double> solution_enriched(dof_handler_enriched.n_dofs());
      radial::recover_solution_ppr(dof_handler, mapping, solution,
                                   dof_handler_enriched, solution_enriched);

      timer.stop();
      std::cout << "  Recovery: " << timer.last_wall_time() << std::endl;
      //-------------------------------------------------------------------------//

      //-------------------------------------------------------------------------//
      // Compute errors for convergence plots
      timer.start();

      QWitherdenVincentSimplex<dim> error_quadrature(order_enriched + 2);

      Vector<double> cell_errors(triangulation.n_cells());

      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        ExactSolution<dim>(),
                                        cell_errors,
                                        error_quadrature,
                                        VectorTools::L2_norm);
      double L2_error = VectorTools::compute_global_error(triangulation,
                                                          cell_errors,
                                                          VectorTools::L2_norm);

      VectorTools::integrate_difference(mapping,
                                        dof_handler_enriched,
                                        solution_enriched,
                                        ExactSolution<dim>(),
                                        cell_errors,
                                        error_quadrature,
                                        VectorTools::L2_norm);
      double L2_error_enriched = VectorTools::compute_global_error(triangulation,
                                                                  cell_errors,
                                                                  VectorTools::L2_norm);

      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        ExactSolution<dim>(),
                                        cell_errors,
                                        error_quadrature,
                                        VectorTools::H1_seminorm);
      double H1_error = VectorTools::compute_global_error(triangulation,
                                                          cell_errors,
                                                          VectorTools::H1_seminorm);

      VectorTools::integrate_difference(mapping,
                                        dof_handler_enriched,
                                        solution_enriched,
                                        ExactSolution<dim>(),
                                        cell_errors,
                                        error_quadrature,
                                        VectorTools::H1_seminorm);
      double H1_error_enriched = VectorTools::compute_global_error(triangulation,
                                                                  cell_errors,
                                                                  VectorTools::H1_seminorm);
      timer.stop();
      std::cout << "  Error   : " << timer.last_wall_time() << std::endl;

      h_vals[level] = 1 / std::pow(triangulation.n_cells(), 1.0/dim);
      L2_vals[level] = L2_error;
      L2E_vals[level] = L2_error_enriched;
      H1_vals[level] = H1_error;
      H1E_vals[level] = H1_error_enriched;

      if (level > 0)
      {
        double delta_h = std::log(h_vals[level]) - std::log(h_vals[level - 1]);
        L2_rates[level] = (std::log(L2_vals[level]) - std::log(L2_vals[level - 1])) / delta_h;
        L2E_rates[level] = (std::log(L2E_vals[level]) - std::log(L2E_vals[level - 1])) / delta_h;
        H1_rates[level] = (std::log(H1_vals[level]) - std::log(H1_vals[level - 1])) / delta_h;
        H1E_rates[level] = (std::log(H1E_vals[level]) - std::log(H1E_vals[level - 1])) / delta_h;
      }

      convergence_table.add_value("level", level);
      convergence_table.add_value("cells", triangulation.n_cells());
      convergence_table.add_value("h_E", 1 / std::pow(triangulation.n_cells(), 1.0/dim));
      convergence_table.add_value("L2", L2_error);
      convergence_table.add_value("L2 rate", L2_rates[level]);
      convergence_table.add_value("L2E", L2_error_enriched);
      convergence_table.add_value("L2E rate", L2E_rates[level]);
      convergence_table.add_value("H1", H1_error);
      convergence_table.add_value("H1 rate", H1_rates[level]);
      convergence_table.add_value("H1E", H1_error_enriched);
      convergence_table.add_value("H1E rate", H1E_rates[level]);
      //-------------------------------------------------------------------------//
    }

    convergence_table.set_precision("h_E", 3);
    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("L2 rate", 3);
    convergence_table.set_precision("L2E", 3);
    convergence_table.set_precision("L2E rate", 3);
    convergence_table.set_precision("H1", 3);
    convergence_table.set_precision("H1 rate", 3);
    convergence_table.set_precision("H1E", 3);
    convergence_table.set_precision("H1E rate", 3);
    convergence_table.set_scientific("h_E", true);
    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("L2E", true);
    convergence_table.set_scientific("H1", true);
    convergence_table.set_scientific("H1E", true);

    std::cout << std::endl;
    convergence_table.write_text(std::cout);
  }
}
