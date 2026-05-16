#ifndef RECOVERY_COMMON_H
#define RECOVERY_COMMON_H

#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>

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
}

#endif // RECOVERY_COMMON_H
