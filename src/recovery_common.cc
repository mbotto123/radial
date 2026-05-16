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
} // namespace radial