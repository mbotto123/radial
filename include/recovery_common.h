#ifndef RECOVERY_COMMON_H
#define RECOVERY_COMMON_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/point.h>

#include <set>

namespace radial
{
  using namespace dealii;

  template <int dim>
  using cell_pointer = typename DoFHandler<dim>::active_cell_iterator;

  template <int dim>
  void create_patch_basis(const unsigned int order,
                          std::vector<std::function<double(Point<dim>)>>& patch_basis_funcs);

  template <int dim>
  void create_vertex_to_cell(const DoFHandler<dim>& dof_handler,
                             const DoFHandler<dim>& dof_handler_enriched,
                             std::vector<std::vector<radial::cell_pointer<dim>>>& vertex_to_cell,
                             std::vector<std::vector<radial::cell_pointer<dim>>>& vertex_to_cell_enriched);

  template<int dim>
  void find_patch_bounding_box(const DoFHandler<dim>& dof_handler,
                               const std::set<unsigned int>& patch_vertices,
                               Point<dim>& coord_min, Point<dim>& coord_max);
  template <int dim>
  void find_patch_bounding_box(const std::vector<radial::cell_pointer<dim>>& patch_cells,
                               const std::set<types::global_dof_index>& patch_dofs,
                               FEValues<dim>& fe_values_nodes,
                               std::vector<types::global_dof_index>& local_dof_indices,
                               Point<dim>& coord_min, Point<dim>& coord_max);
  
  template <int dim>
  unsigned int get_min_points(const unsigned int order_enriched);

  template<int dim>
  void least_squares_patch(const std::vector<radial::cell_pointer<dim>>& patch_cells,
                           const std::set<types::global_dof_index>& patch_dofs,
                           const Point<dim>& patch_coord_min,
                           const Point<dim>& patch_coord_max,
                           const std::vector<std::function<double(Point<dim>)>>& patch_basis_funcs,
                           const FiniteElement<dim>& fe,
                           const Vector<double>& solution,
                           FEValues<dim>& fe_values_nodes,
                           Vector<double>& lsq_coeffs,
                           double& rcond);
}

#endif // RECOVERY_COMMON_H
