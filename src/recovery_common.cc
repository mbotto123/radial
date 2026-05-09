#include "recovery_common_impl.h"

namespace radial
{
  using namespace dealii;

  // Explicit instantiations
  template void create_patch_basis<2>(const unsigned int order,
                                      std::vector<std::function<double(Point<2>)>>& patch_basis_funcs);

  template void create_patch_basis<3>(const unsigned int order,
                                      std::vector<std::function<double(Point<3>)>>& patch_basis_funcs);
} // namespace radial