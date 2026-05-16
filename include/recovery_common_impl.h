#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/point.h>
#include <deal.II/base/table.h>

#include <functional>

// Uncomment if adding Legendre basis option later
// #include <gsl/gsl_sf_legendre.h>

// Common functions used by recovery methods

namespace radial
{
  using namespace dealii;

  // Fill data structure that contains the baseline patch for every vertex.
  //
  // The way a "patch" is implemented is as a list of iterators, i.e. pointers
  // to the different cells in the patch. These pointers may be pointing to
  // cells that are far away from each other in memory, so maybe there is a
  // more efficient way to implement this in terms of memory access. TODO.
  //
  // Two data structures are filled, one for the non-enriched field, and one
  // for the enriched field.
  template <int dim>
  void create_vertex_to_cell(const DoFHandler<dim>& dof_handler,
                             const DoFHandler<dim>& dof_handler_enriched,
                             std::vector<std::list<typename DoFHandler<dim>::active_cell_iterator>>& vertex_to_cell,
                             std::vector<std::list<typename DoFHandler<dim>::active_cell_iterator>>& vertex_to_cell_enriched)
  {
    const Triangulation<dim>& triangulation = dof_handler.get_triangulation();

    vertex_to_cell.resize(triangulation.n_vertices());
    vertex_to_cell_enriched.resize(triangulation.n_vertices());

    // Get iterator for enriched field explicitly. We need to take care of incrementing
    // this iterator manually, so that it keeps up with the iterator we're looping over.
    typename DoFHandler<dim>::active_cell_iterator cell_enriched_it = dof_handler_enriched.begin();

    for (const auto &cell: dof_handler.active_cell_iterators())
    {
      for (const auto v: cell->vertex_indices())
      {
        // Add base field cell
        vertex_to_cell[cell->vertex_index(v)].emplace_back(cell);
        // Add enriched field cell
        vertex_to_cell_enriched[cell->vertex_index(v)].emplace_back(cell_enriched_it);
      }
      
      ++cell_enriched_it; // This iterator needs to be incremented manually
    }
  }

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

    // Testing Legendre basis. This didn't end up helping with conditioning
    // when the patch was too small, but leaving it commented in here so that
    // it can be potentially incorporated as an option.
    /*
    const unsigned int basis_order = order + 1;

    if (dim == 2)
    {
      const unsigned int basis_size = 0.5 * (basis_order + 1) * (basis_order + 2);

      if (order == 1)
      {
        const int leg_indices_array[] = {0, 0,
                                         1, 0,
                                         0, 1,
                                         2, 0,
                                         1, 1,
                                         0, 2};
        Table<2, int> leg_indices(basis_size, dim, leg_indices_array);

        for (unsigned int i = 0; i < basis_size; i++)
        {
          // TODO: Table can't be captured by reference? Try to replace with
          // something else that can be captured by reference?
          patch_basis_funcs[i] = [leg_indices, i](Point<dim> psi){
            double basis_term = 1;
            for (int d = 0; d < dim; d++)
            {
              basis_term *= gsl_sf_legendre_Pl(leg_indices[i][d], psi(d));
            }
            return basis_term;
          };
        }
      }
      else if (order == 2)
      {
        const int leg_indices_array[] = {0, 0,
                                         1, 0,
                                         0, 1,
                                         2, 0,
                                         1, 1,
                                         0, 2,
                                         3, 0,
                                         2, 1,
                                         1, 2,
                                         0, 3};
        Table<2, int> leg_indices(basis_size, dim, leg_indices_array);

        for (unsigned int i = 0; i < basis_size; i++)
        {
          // TODO: Table can't be captured by reference? Try to replace with
          // something else that can be captured by reference?
          patch_basis_funcs[i] = [leg_indices, i](Point<dim> psi){
            double basis_term = 1;
            for (int d = 0; d < dim; d++)
            {
              basis_term *= gsl_sf_legendre_Pl(leg_indices[i][d], psi(d));
            }
            return basis_term;
          };
        }
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
      const unsigned int basis_size = (1.0/6.0) *
                                      (basis_order + 1) *
                                      (basis_order + 2) *
                                      (basis_order + 3);

      if (order == 1)
      {
        const int leg_indices_array[] = {0, 0, 0,
                                         1, 0, 0,
                                         0, 1, 0,
                                         0, 0, 1,
                                         2, 0, 0,
                                         1, 1, 0,
                                         1, 0, 1,
                                         0, 2, 0,
                                         0, 1, 1,
                                         0, 0, 2};
        Table<2, int> leg_indices(basis_size, dim, leg_indices_array);

        for (unsigned int i = 0; i < basis_size; i++)
        {
          // TODO: Table can't be captured by reference? Try to replace with
          // something else that can be captured by reference?
          patch_basis_funcs[i] = [leg_indices, i](Point<dim> psi){
            double basis_term = 1;
            for (int d = 0; d < dim; d++)
            {
              basis_term *= gsl_sf_legendre_Pl(leg_indices[i][d], psi(d));
            }
            return basis_term;
          };
        }
      }
      else if (order == 2)
      {
        const int leg_indices_array[] = {0, 0, 0,
                                         1, 0, 0,
                                         0, 1, 0,
                                         0, 0, 1,
                                         2, 0, 0,
                                         1, 1, 0,
                                         1, 0, 1,
                                         0, 2, 0,
                                         0, 1, 1,
                                         0, 0, 2,
                                         3, 0, 0,
                                         2, 1, 0,
                                         2, 0, 1,
                                         1, 2, 0,
                                         1, 1, 1,
                                         1, 0, 2,
                                         0, 3, 0,
                                         0, 2, 1,
                                         0, 1, 2,
                                         0, 0, 3};
        Table<2, int> leg_indices(basis_size, dim, leg_indices_array);

        for (unsigned int i = 0; i < basis_size; i++)
        {
          // TODO: Table can't be captured by reference? Try to replace with
          // something else that can be captured by reference?
          patch_basis_funcs[i] = [leg_indices, i](Point<dim> psi){
            double basis_term = 1;
            for (int d = 0; d < dim; d++)
            {
              basis_term *= gsl_sf_legendre_Pl(leg_indices[i][d], psi(d));
            }
            return basis_term;
          };
        }
      }
      else
      {
        // deal.ii does not currently support >P3 simplices, so we cannot do
        // P3 to P4 enrichment
        Assert(order <= 2,
               ExcMessage("Recovery not possible beyond P2 because deal.ii doesn't support >P3 simplices yet."));
      }
    }
    */
  }

  // Find the bounding box of a patch of cells.
  //
  // Implemented by finding the minimum and maximum physical coordinates over
  // all nodes in the patch. Note that we use nodes rather than vertices so that
  // this computation will be valid for curved meshes as well as linear meshes.
  // On a curved mesh, it's possible that the minimum/maximum coordinates will
  // come from an edge node rather than a vertex.
  template <int dim>
  void find_patch_bounding_box(const std::set<typename DoFHandler<dim>::active_cell_iterator>& patch_cells,
                               const std::set<types::global_dof_index>& patch_dofs,
                               FEValues<dim>& fe_values_nodes,
                               std::vector<types::global_dof_index>& local_dof_indices,
                               Point<dim>& coord_min, Point<dim>& coord_max)
  {
    std::vector<std::vector<double>> coord_patch_nodes(dim);
    for (int d = 0; d < dim; d++)
      coord_patch_nodes[d].resize(patch_dofs.size());

    std::set<types::global_dof_index> traversed_nodes;
    unsigned int node_count = 0;

    // Loop over patch cells and get physical coordinates of the nodes
    for (const auto &cell: patch_cells)
    {
      fe_values_nodes.reinit(cell);

      cell->get_dof_indices(local_dof_indices);

      for (const unsigned int i : fe_values_nodes.quadrature_point_indices())
      {
        if (traversed_nodes.count(local_dof_indices[i]) < 1) // if we haven't been to this node yet
        {
          Point<dim> node_physical_coords = fe_values_nodes.quadrature_point(i);

          for (int d = 0; d < dim; d++)
            coord_patch_nodes[d][node_count] = node_physical_coords(d);

          node_count++;
        }
        traversed_nodes.insert(local_dof_indices[i]);
      }
    }

    // Find limits of the bounding box that contains the patch
    for (int d = 0; d < dim; d++)
    {
      coord_min(d) = *std::min_element(coord_patch_nodes[d].begin(),
                                       coord_patch_nodes[d].end());
      coord_max(d) = *std::max_element(coord_patch_nodes[d].begin(),
                                       coord_patch_nodes[d].end());
    }
  }

  // Minimum number of sampling points required to get a solvable system on a
  // patch. This is the number required for interpolation. Least-squares will
  // need at least 1 more than this.
  //
  // In a more general context, this is just the number of coefficients (or
  // linearly independent basis functions) needed to define a polynomial with
  // the given order.
  template <int dim>
  unsigned int get_min_points(const unsigned int order_enriched)
  {
    unsigned int min_points = 1;

    double inverse_factorial = 1;
    for (int d = 1; d <= dim; d++)
    {
      inverse_factorial /= d;
      min_points *= (order_enriched + d);
    }
    min_points *= inverse_factorial;

    return min_points;
  }
} // namespace radial