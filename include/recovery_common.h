#ifndef RECOVERY_COMMON_H
#define RECOVERY_COMMON_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/point.h>

namespace radial
{
  using namespace dealii;

  template <int dim>
  void create_patch_basis(const unsigned int order,
                          std::vector<std::function<double(Point<dim>)>>& patch_basis_funcs);

  template <int dim>
  void create_vertex_to_cell(const DoFHandler<dim>& dof_handler,
                             const DoFHandler<dim>& dof_handler_enriched,
                             std::vector<std::list<typename DoFHandler<dim>::active_cell_iterator>>& vertex_to_cell,
                             std::vector<std::list<typename DoFHandler<dim>::active_cell_iterator>>& vertex_to_cell_enriched);

  template <int dim>
  void find_patch_bounding_box(const std::set<typename DoFHandler<dim>::active_cell_iterator>& patch_cells,
                               const std::set<types::global_dof_index>& patch_dofs,
                               FEValues<dim>& fe_values_nodes,
                               std::vector<types::global_dof_index>& local_dof_indices,
                               Point<dim>& coord_min, Point<dim>& coord_max);
}

#endif // RECOVERY_COMMON_H
