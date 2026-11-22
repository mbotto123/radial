#include "recovery/recovery_discrete_impl.h"

namespace radial
{
  using namespace dealii;

  // Explicit instantiations

  template void
  least_squares_patch<2>(const std::vector<radial::cell_pointer<2>>& patch_cells,
                         const std::set<types::global_dof_index>& patch_dofs,
                         const Point<2>& patch_coord_min,
                         const Point<2>& patch_coord_max,
                         const radial::patch_basis<2>& patch_basis_funcs,
                         const FiniteElement<2>& fe,
                         const Vector<double>& solution,
                         FEValues<2>& fe_values_nodes,
                         Vector<double>& lsq_coeffs,
                         double& rcond);
  template void
  least_squares_patch<3>(const std::vector<radial::cell_pointer<3>>& patch_cells,
                         const std::set<types::global_dof_index>& patch_dofs,
                         const Point<3>& patch_coord_min,
                         const Point<3>& patch_coord_max,
                         const radial::patch_basis<3>& patch_basis_funcs,
                         const FiniteElement<3>& fe,
                         const Vector<double>& solution,
                         FEValues<3>& fe_values_nodes,
                         Vector<double>& lsq_coeffs,
                         double& rcond);

  template void
  least_squares_patch_discrete<2>(const std::vector<Point<2>>& dof_coords,
                                  const Vector<double>& solution,
                                  const std::set<types::global_dof_index>& patch_dofs,
                                  const unsigned int& min_points,
                                  const Point<2>& patch_coord_min,
                                  const Point<2>& patch_coord_max,
                                  const radial::patch_basis<2>& patch_basis_funcs,
                                  Vector<double>& lsq_coeffs,
                                  double& rcond);
  template void
  least_squares_patch_discrete<3>(const std::vector<Point<3>>& dof_coords,
                                  const Vector<double>& solution,
                                  const std::set<types::global_dof_index>& patch_dofs,
                                  const unsigned int& min_points,
                                  const Point<3>& patch_coord_min,
                                  const Point<3>& patch_coord_max,
                                  const radial::patch_basis<3>& patch_basis_funcs,
                                  Vector<double>& lsq_coeffs,
                                  double& rcond);

  template void
  recover_solution_ppr_discrete<2>(const DoFHandler<2>& dof_handler,
                                   const MappingP1<2>& mapping,
                                   const Vector<double>& solution,
                                   const DoFHandler<2>& dof_handler_enriched,
                                   Vector<double>& solution_enriched);
  template void
  recover_solution_ppr_discrete<3>(const DoFHandler<3>& dof_handler,
                                   const MappingP1<3>& mapping,
                                   const Vector<double>& solution,
                                   const DoFHandler<3>& dof_handler_enriched,
                                   Vector<double>& solution_enriched);
} // namespace radial
