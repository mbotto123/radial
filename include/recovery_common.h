#ifndef RECOVERY_COMMON_H
#define RECOVERY_COMMON_H

#include <deal.II/base/point.h>

namespace radial
{
  using namespace dealii;

  template <int dim>
  void create_patch_basis(const unsigned int order,
                          std::vector<std::function<double(Point<dim>)>>& patch_basis_funcs);
}

#endif // RECOVERY_COMMON_H
