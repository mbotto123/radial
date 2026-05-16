#include <iostream>

#include <recovery_common.h>

//----------------------------------------------------------------------------//
// Test of computing minimum number of coefficients to define a polynomial in 3D
//----------------------------------------------------------------------------//

void min_points_test_3D(const unsigned int order_enriched)
{
  const int dim = 3;

  unsigned int min_points = radial::get_min_points<dim>(order_enriched);

  std::cout << min_points << std::endl;
}

int main()
{
  min_points_test_3D(2);
  min_points_test_3D(3);
}