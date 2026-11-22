#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_p1.h>

#include <deal.II/lac/vector.h>

#include <recovery_discrete.h>
#include <recovery_integral.h>

namespace radial
{
  using namespace dealii;

  template <int dim>
  void recover_solution(const DoFHandler<dim>& dof_handler, const MappingP1<dim>& mapping,
                        const Vector<double>& solution,
                        const DoFHandler<dim>& dof_handler_enriched,
                        Vector<double>& solution_enriched,
                        const bool use_integral_least_squares = false)
  {
    if (use_integral_least_squares)
    {
      recover_solution_ppr_integral(dof_handler, mapping, solution,
                                    dof_handler_enriched, solution_enriched);
    }
    else
    {
      recover_solution_ppr_discrete(dof_handler, mapping, solution,
                                    dof_handler_enriched, solution_enriched);
    }
  }
} // namespace radial
