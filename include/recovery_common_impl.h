#include <deal.II/base/point.h>

// Common functions used by recovery methods

namespace radial
{
  using namespace dealii;

  // Create a set of basis functions representing a global polynomial over a
  // patch of elements, to be used for a least-squares problem on the patch.
  template <int dim>
  void create_patch_basis(const unsigned int order,
                          std::vector<std::function<double(Point<dim>)>>& patch_basis_funcs)
  {
    if (dim == 2)
    {
      if (order == 1)
      {
        patch_basis_funcs[0] = [](Point<dim> psi){ return 1.0; };
        patch_basis_funcs[1] = [](Point<dim> psi){ return psi(0); };
        patch_basis_funcs[2] = [](Point<dim> psi){ return psi(1); };
        patch_basis_funcs[3] = [](Point<dim> psi){ return psi(0)*psi(0); };
        patch_basis_funcs[4] = [](Point<dim> psi){ return psi(0)*psi(1); };
        patch_basis_funcs[5] = [](Point<dim> psi){ return psi(1)*psi(1); };
      }
      else if (order == 2)
      {
        patch_basis_funcs[0] = [](Point<dim> psi){ return 1.0; };
        patch_basis_funcs[1] = [](Point<dim> psi){ return psi(0); };
        patch_basis_funcs[2] = [](Point<dim> psi){ return psi(1); };
        patch_basis_funcs[3] = [](Point<dim> psi){ return psi(0)*psi(0); };
        patch_basis_funcs[4] = [](Point<dim> psi){ return psi(0)*psi(1); };
        patch_basis_funcs[5] = [](Point<dim> psi){ return psi(1)*psi(1); };
        patch_basis_funcs[6] = [](Point<dim> psi){ return psi(0)*psi(0)*psi(0); };
        patch_basis_funcs[7] = [](Point<dim> psi){ return psi(0)*psi(0)*psi(1); };
        patch_basis_funcs[8] = [](Point<dim> psi){ return psi(0)*psi(1)*psi(1); };
        patch_basis_funcs[9] = [](Point<dim> psi){ return psi(1)*psi(1)*psi(1); };
      }
      else
      {
        // deal.ii does not currently support >P3 simplices, so we cannot do
        // P3 to P4 enrichment
        Assert(order <= 2,
               ExcMessage("Recovery not possible beyond P2 because deal.ii doesn't support >P3 simplices yet."));
      }
    }
    else if (dim == 3)
    {
      if (order == 1)
      {
        patch_basis_funcs[0] = [](Point<dim> psi){ return 1.0; };
        patch_basis_funcs[1] = [](Point<dim> psi){ return psi(0); };
        patch_basis_funcs[2] = [](Point<dim> psi){ return psi(1); };
        patch_basis_funcs[3] = [](Point<dim> psi){ return psi(2); };
        patch_basis_funcs[4] = [](Point<dim> psi){ return psi(0)*psi(0); };
        patch_basis_funcs[5] = [](Point<dim> psi){ return psi(0)*psi(1); };
        patch_basis_funcs[6] = [](Point<dim> psi){ return psi(0)*psi(2); };
        patch_basis_funcs[7] = [](Point<dim> psi){ return psi(1)*psi(1); };
        patch_basis_funcs[8] = [](Point<dim> psi){ return psi(1)*psi(2); };
        patch_basis_funcs[9] = [](Point<dim> psi){ return psi(2)*psi(2); };
      }
      else if (order == 2)
      {
        patch_basis_funcs[0]  = [](Point<dim> psi){ return 1.0; };
        patch_basis_funcs[1]  = [](Point<dim> psi){ return psi(0); };
        patch_basis_funcs[2]  = [](Point<dim> psi){ return psi(1); };
        patch_basis_funcs[3]  = [](Point<dim> psi){ return psi(2); };
        patch_basis_funcs[4]  = [](Point<dim> psi){ return psi(0)*psi(0); };
        patch_basis_funcs[5]  = [](Point<dim> psi){ return psi(0)*psi(1); };
        patch_basis_funcs[6]  = [](Point<dim> psi){ return psi(0)*psi(2); };
        patch_basis_funcs[7]  = [](Point<dim> psi){ return psi(1)*psi(1); };
        patch_basis_funcs[8]  = [](Point<dim> psi){ return psi(1)*psi(2); };
        patch_basis_funcs[9]  = [](Point<dim> psi){ return psi(2)*psi(2); };
        patch_basis_funcs[10] = [](Point<dim> psi){ return psi(0)*psi(0)*psi(0); };
        patch_basis_funcs[11] = [](Point<dim> psi){ return psi(0)*psi(0)*psi(1); };
        patch_basis_funcs[12] = [](Point<dim> psi){ return psi(0)*psi(0)*psi(2); };
        patch_basis_funcs[13] = [](Point<dim> psi){ return psi(0)*psi(1)*psi(1); };
        patch_basis_funcs[14] = [](Point<dim> psi){ return psi(0)*psi(1)*psi(2); };
        patch_basis_funcs[15] = [](Point<dim> psi){ return psi(0)*psi(2)*psi(2); };
        patch_basis_funcs[16] = [](Point<dim> psi){ return psi(1)*psi(1)*psi(1); };
        patch_basis_funcs[17] = [](Point<dim> psi){ return psi(1)*psi(1)*psi(2); };
        patch_basis_funcs[18] = [](Point<dim> psi){ return psi(1)*psi(2)*psi(2); };
        patch_basis_funcs[19] = [](Point<dim> psi){ return psi(2)*psi(2)*psi(2); };
      }
      else
      {
        // deal.ii does not currently support >P3 simplices, so we cannot do
        // P3 to P4 enrichment
        Assert(order <= 2,
               ExcMessage("Recovery not possible beyond P2 because deal.ii doesn't support >P3 simplices yet."));
      }
    }
  }
} // namespace radial