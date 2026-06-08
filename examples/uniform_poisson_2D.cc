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

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/convergence_table.h>

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
  //-------------------------------------------------------------------------//

  for (int order = min_order; order <= max_order; order++)
  {
    std::cout << std::endl << "Running P" << order << " case..." << std::endl;

    ConvergenceTable convergence_table;

    for (int level = 0; level < max_level; level++)
    {
      //-------------------------------------------------------------------------//
      // Mesh
      int repetitions = min_repetitions * std::pow(std::sqrt(2.0), level);

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
      // Allocate memory for system matrix, RHS, and solution
      SparsityPattern sparsity_pattern;
      DynamicSparsityPattern dsp(dof_handler.n_dofs());

      DoFTools::make_sparsity_pattern(dof_handler, dsp);
      sparsity_pattern.copy_from(dsp);

      SparseMatrix<double> system_matrix(sparsity_pattern);
      Vector<double> system_rhs(dof_handler.n_dofs());
      Vector<double> solution(dof_handler.n_dofs());
      //-------------------------------------------------------------------------//

      //-------------------------------------------------------------------------//
      // Construct Poisson matrix and RHS
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
      //-------------------------------------------------------------------------//

      //-------------------------------------------------------------------------//
      // Solve system
      SparseDirectUMFPACK A_direct;

      solution = system_rhs;
      A_direct.solve(system_matrix, solution);
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
      Vector<double> solution_enriched(dof_handler_enriched.n_dofs());
      radial::recover_solution_ppr(dof_handler, mapping, solution,
                                   dof_handler_enriched, solution_enriched);
      //-------------------------------------------------------------------------//

      //-------------------------------------------------------------------------//
      // Compute errors for convergence plots
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

      convergence_table.add_value("level", level);
      convergence_table.add_value("cells", triangulation.n_cells());
      convergence_table.add_value("h_E", 1 / std::pow(triangulation.n_cells(), 1.0/dim));
      convergence_table.add_value("L2", L2_error);
      convergence_table.add_value("L2E", L2_error_enriched);
      convergence_table.add_value("H1", H1_error);
      convergence_table.add_value("H1E", H1_error_enriched);
      //-------------------------------------------------------------------------//
    }

    convergence_table.set_precision("h_E", 3);
    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("L2E", 3);
    convergence_table.set_precision("H1", 3);
    convergence_table.set_precision("H1E", 3);
    convergence_table.set_scientific("h_E", true);
    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("L2E", true);
    convergence_table.set_scientific("H1", true);
    convergence_table.set_scientific("H1E", true);

    std::cout << std::endl;
    convergence_table.write_text(std::cout);
  }
}
