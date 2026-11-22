#ifndef RECOVERY_INTEGRAL_H
#define RECOVERY_INTEGRAL_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_p1.h>
#include <deal.II/lac/vector.h>

#include <recovery/recovery_common.h>

namespace radial
{
  using namespace dealii;

  template <int dim>
  void create_vertex_to_cell(const DoFHandler<dim>& dof_handler,
                             std::vector<std::vector<radial::cell_pointer<dim>>>& vertex_to_cell);

  template<int dim>
  void least_squares_patch_integral(const std::vector<radial::cell_pointer<dim>>& patch_cells,
                                    const Point<dim>& patch_coord_min,
                                    const Point<dim>& patch_coord_max,
                                    const radial::patch_basis<dim>& patch_basis_funcs,
                                    const MappingP1<dim>& mapping,
                                    const FiniteElement<dim>& fe,
                                    const Vector<double>& solution,
                                    Vector<double>& lsq_coeffs);

  template <int dim>
  void recover_solution_ppr_integral(const DoFHandler<dim>& dof_handler,
                                     const MappingP1<dim>& mapping,
                                     const Vector<double>& solution,
                                     const DoFHandler<dim>& dof_handler_enriched,
                                     Vector<double>& solution_enriched);
} // namespace radial

#endif // RECOVERY_INTEGRAL_H
