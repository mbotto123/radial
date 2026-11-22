#include <deal.II/fe/mapping_p1.h>

#include <set>

#include "recovery_common_impl.h"

namespace radial
{
  using namespace dealii;

  // Explicit instantiations

  template void
  create_vertex_mappings<2>(const DoFHandler<2>& dof_handler,
                            const DoFHandler<2>& dof_handler_enriched,
                            std::vector<std::set<types::global_vertex_index>>& vertex_to_vertex,
                            std::vector<std::set<types::global_dof_index>>& vertex_to_dof,
                            std::vector<std::vector<types::global_dof_index>>& vertex_to_dof_enriched,
                            std::vector<std::vector<double>>& vertex_to_weight);
  template void
  create_vertex_mappings<3>(const DoFHandler<3>& dof_handler,
                            const DoFHandler<3>& dof_handler_enriched,
                            std::vector<std::set<types::global_vertex_index>>& vertex_to_vertex,
                            std::vector<std::set<types::global_dof_index>>& vertex_to_dof,
                            std::vector<std::vector<types::global_dof_index>>& vertex_to_dof_enriched,
                            std::vector<std::vector<double>>& vertex_to_weight);

  template void create_patch_basis<2>(const unsigned int order,
                                      radial::patch_basis<2>& patch_basis_funcs);
  template void create_patch_basis<3>(const unsigned int order,
                                      radial::patch_basis<3>& patch_basis_funcs);

  template void
  find_patch_bounding_box<2>(const DoFHandler<2>& dof_handler,
                             const std::set<types::global_vertex_index>& patch_vertices,
                             Point<2>& coord_min, Point<2>& coord_max);
  template void
  find_patch_bounding_box<3>(const DoFHandler<3>& dof_handler,
                             const std::set<types::global_vertex_index>& patch_vertices,
                             Point<3>& coord_min, Point<3>& coord_max);
  template void
  find_patch_bounding_box<2>(const std::vector<Point<2>>& dof_coords,
                             const std::set<types::global_dof_index>& patch_dofs,
                             Point<2>& coord_min, Point<2>& coord_max);
  template void
  find_patch_bounding_box<3>(const std::vector<Point<3>>& dof_coords,
                             const std::set<types::global_dof_index>& patch_dofs,
                             Point<3>& coord_min, Point<3>& coord_max);
  template void
  find_patch_bounding_box<2>(const std::vector<radial::cell_pointer<2>>& patch_cells,
                             const std::set<types::global_dof_index>& patch_dofs,
                             FEValues<2>& fe_values_nodes,
                             std::vector<types::global_dof_index>& local_dof_indices,
                             Point<2>& coord_min, Point<2>& coord_max);
  template void
  find_patch_bounding_box<3>(const std::vector<radial::cell_pointer<3>>& patch_cells,
                             const std::set<types::global_dof_index>& patch_dofs,
                             FEValues<3>& fe_values_nodes,
                             std::vector<types::global_dof_index>& local_dof_indices,
                             Point<3>& coord_min, Point<3>& coord_max);

  template unsigned int get_min_points<2>(const unsigned int order_enriched);
  template unsigned int get_min_points<3>(const unsigned int order_enriched);

  template void
  evaluate_patch_polynomial<2>(const std::vector<Point<2>>& dof_coords_enriched,
                               const std::vector<types::global_dof_index>& patch_dofs_enriched,
                               const std::vector<double>& patch_weights,
                               const Point<2>& patch_coord_min,
                               const Point<2>& patch_coord_max,
                               const radial::patch_basis<2>& patch_basis_funcs,
                               const Vector<double>& lsq_coeffs,
                               Vector<double>& solution_enriched);
  template void
  evaluate_patch_polynomial<3>(const std::vector<Point<3>>& dof_coords_enriched,
                               const std::vector<types::global_dof_index>& patch_dofs_enriched,
                               const std::vector<double>& patch_weights,
                               const Point<3>& patch_coord_min,
                               const Point<3>& patch_coord_max,
                               const radial::patch_basis<3>& patch_basis_funcs,
                               const Vector<double>& lsq_coeffs,
                               Vector<double>& solution_enriched);
} // namespace radial