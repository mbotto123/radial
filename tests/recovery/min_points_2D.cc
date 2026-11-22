#include <iostream>

#include <recovery_common.h>

//----------------------------------------------------------------------------//
// Test of computing minimum number of coefficients to define a polynomial in 2D
//----------------------------------------------------------------------------//

void min_points_test_2D(const unsigned int order_enriched)
{
  const int dim = 2;

  unsigned int min_points = radial::get_min_points<dim>(order_enriched);

  std::cout << min_points << std::endl;
}

int main()
{
  min_points_test_2D(2);
  min_points_test_2D(3);
}