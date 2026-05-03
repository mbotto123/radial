#ifndef SOLUTION_RECOVERY_H
#define SOLUTION_RECOVERY_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_p1.h>
#include <deal.II/lac/vector.h>

namespace radial
{
  using namespace dealii;

  template <int dim>
  void recover_solution_ppr(const DoFHandler<dim>& dof_handler, const MappingP1<dim>& mapping,
                            const Vector<double>& solution,
                            const DoFHandler<dim>& dof_handler_enriched,
                            Vector<double>& solution_enriched);
} // namespace radial

#endif // SOLUTION_RECOVERY_H
