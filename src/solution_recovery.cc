#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_p1.h>
#include <deal.II/lac/vector.h>

#include "solution_recovery_impl.h"

namespace radial
{
  using namespace dealii;

  // Explicit instantiations
  template void recover_solution_ppr<2>(const DoFHandler<2>& dof_handler, const MappingP1<2>& mapping,
                                        const Vector<double>& solution,
                                        const DoFHandler<2>& dof_handler_enriched,
                                        Vector<double>& solution_enriched);
} // namespace radial
