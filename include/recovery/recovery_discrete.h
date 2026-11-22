#ifndef RECOVERY_DISCRETE_H
#define RECOVERY_DISCRETE_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_p1.h>
#include <deal.II/lac/vector.h>

#include <recovery/recovery_common.h>

namespace radial
{
  using namespace dealii;

  template<int dim>
  void least_squares_patch(const std::vector<radial::cell_pointer<dim>>& patch_cells,
                           const std::set<types::global_dof_index>& patch_dofs,
                           const Point<dim>& patch_coord_min,
                           const Point<dim>& patch_coord_max,
                           const radial::patch_basis<dim>& patch_basis_funcs,
                           const FiniteElement<dim>& fe,
                           const Vector<double>& solution,
                           FEValues<dim>& fe_values_nodes,
                           Vector<double>& lsq_coeffs,
                           double& rcond);

  template<int dim>
  void least_squares_patch_discrete(const std::vector<Point<dim>>& dof_coords,
                                    const Vector<double>& solution,
                                    const std::set<types::global_dof_index>& patch_dofs,
                                    const unsigned int& min_points,
                                    const Point<dim>& patch_coord_min,
                                    const Point<dim>& patch_coord_max,
                                    const radial::patch_basis<dim>& patch_basis_funcs,
                                    Vector<double>& lsq_coeffs,
                                    double& rcond);

  template <int dim>
  void recover_solution_ppr_discrete(const DoFHandler<dim>& dof_handler,
                                     const MappingP1<dim>& mapping,
                                     const Vector<double>& solution,
                                     const DoFHandler<dim>& dof_handler_enriched,
                                     Vector<double>& solution_enriched);
} // namespace radial

#endif // RECOVERY_DISCRETE_H
