#include "recovery_integral_impl.h"

namespace radial
{
  using namespace dealii;

  // Explicit instantiations

  template void create_vertex_to_cell<2>(const DoFHandler<2>& dof_handler,
                                         std::vector<std::vector<radial::cell_pointer<2>>>& vertex_to_cell);
  template void create_vertex_to_cell<3>(const DoFHandler<3>& dof_handler,
                                         std::vector<std::vector<radial::cell_pointer<3>>>& vertex_to_cell);

  template void
  least_squares_patch_integral<2>(const std::vector<radial::cell_pointer<2>>& patch_cells,
                                  const Point<2>& patch_coord_min,
                                  const Point<2>& patch_coord_max,
                                  const radial::patch_basis<2>& patch_basis_funcs,
                                  const MappingP1<2>& mapping,
                                  const FiniteElement<2>& fe,
                                  const Vector<double>& solution,
                                  Vector<double>& lsq_coeffs);
  template void
  least_squares_patch_integral<3>(const std::vector<radial::cell_pointer<3>>& patch_cells,
                                  const Point<3>& patch_coord_min,
                                  const Point<3>& patch_coord_max,
                                  const radial::patch_basis<3>& patch_basis_funcs,
                                  const MappingP1<3>& mapping,
                                  const FiniteElement<3>& fe,
                                  const Vector<double>& solution,
                                  Vector<double>& lsq_coeffs);

  template void
  recover_solution_ppr_integral<2>(const DoFHandler<2>& dof_handler,
                                   const MappingP1<2>& mapping,
                                   const Vector<double>& solution,
                                   const DoFHandler<2>& dof_handler_enriched,
                                   Vector<double>& solution_enriched);
  template void
  recover_solution_ppr_integral<3>(const DoFHandler<3>& dof_handler,
                                   const MappingP1<3>& mapping,
                                   const Vector<double>& solution,
                                   const DoFHandler<3>& dof_handler_enriched,
                                   Vector<double>& solution_enriched);
} // namespace radial
