#include "recovery_common_impl.h"

namespace radial
{
  using namespace dealii;

  // Explicit instantiations
  template void create_vertex_to_cell<2>(const DoFHandler<2>& dof_handler,
                                         const DoFHandler<2>& dof_handler_enriched,
                                         std::vector<std::list<typename DoFHandler<2>::active_cell_iterator>>& vertex_to_cell,
                                         std::vector<std::list<typename DoFHandler<2>::active_cell_iterator>>& vertex_to_cell_enriched);
  template void create_vertex_to_cell<3>(const DoFHandler<3>& dof_handler,
                                         const DoFHandler<3>& dof_handler_enriched,
                                         std::vector<std::list<typename DoFHandler<3>::active_cell_iterator>>& vertex_to_cell,
                                         std::vector<std::list<typename DoFHandler<3>::active_cell_iterator>>& vertex_to_cell_enriched);

  template void create_patch_basis<2>(const unsigned int order,
                                      std::vector<std::function<double(Point<2>)>>& patch_basis_funcs);
  template void create_patch_basis<3>(const unsigned int order,
                                      std::vector<std::function<double(Point<3>)>>& patch_basis_funcs);

  template void
  find_patch_bounding_box<2>(const std::set<typename DoFHandler<2>::active_cell_iterator>& patch_cells,
                             const std::set<types::global_dof_index>& patch_dofs,
                             FEValues<2>& fe_values_nodes,
                             std::vector<types::global_dof_index>& local_dof_indices,
                             Point<2>& coord_min, Point<2>& coord_max);
  template void
  find_patch_bounding_box<3>(const std::set<typename DoFHandler<3>::active_cell_iterator>& patch_cells,
                             const std::set<types::global_dof_index>& patch_dofs,
                             FEValues<3>& fe_values_nodes,
                             std::vector<types::global_dof_index>& local_dof_indices,
                             Point<3>& coord_min, Point<3>& coord_max);

  template unsigned int get_min_points<2>(const unsigned int order_enriched);
  template unsigned int get_min_points<3>(const unsigned int order_enriched);
} // namespace radial