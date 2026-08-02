#include <recovery_common.h>

#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/base/point.h>
#include <deal.II/base/table.h>

#include <functional>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

// Uncomment if adding Legendre basis option later
// #include <gsl/gsl_sf_legendre.h>

// Common functions used by recovery methods

namespace radial
{
  using namespace dealii;

  // Fill data structure that contains the baseline patch for every vertex, i.e.
  // all of the cells that are directly attached to a vertex.
  //
  // The way a "patch" is implemented is as a list of iterators, i.e. pointers
  // to the different cells in the patch. These pointers may be pointing to
  // cells that are far away from each other in memory, so maybe there is a
  // more efficient way to implement this in terms of memory access. TODO.
  //
  // Two data structures are filled, one for the non-enriched field, and one
  // for the enriched field. Although the global element indices are the same
  // for both fields, a pointer to a non-enriched field cell is not the same
  // as a pointer to an enriched field cell, so we keep track of them separately.
  template <int dim>
  void create_vertex_to_cell(const DoFHandler<dim>& dof_handler,
                             const DoFHandler<dim>& dof_handler_enriched,
                             std::vector<std::vector<radial::cell_pointer<dim>>>& vertex_to_cell,
                             std::vector<std::vector<radial::cell_pointer<dim>>>& vertex_to_cell_enriched)
  {
    const Triangulation<dim>& triangulation = dof_handler.get_triangulation();

    vertex_to_cell.resize(triangulation.n_vertices());
    vertex_to_cell_enriched.resize(triangulation.n_vertices());

    // Get iterator for enriched field explicitly. We need to take care of incrementing
    // this iterator manually, so that it keeps up with the iterator we're looping over.
    radial::cell_pointer<dim> cell_enriched_it = dof_handler_enriched.begin();

    for (const auto &cell: dof_handler.active_cell_iterators())
    {
      for (const auto v: cell->vertex_indices())
      {
        // Add base field cell
        vertex_to_cell[cell->vertex_index(v)].push_back(cell);
        // Add enriched field cell
        vertex_to_cell_enriched[cell->vertex_index(v)].push_back(cell_enriched_it);
      }
      
      ++cell_enriched_it; // This iterator needs to be incremented manually
    }
  }

  // Fill data structures containing a mapping from each vertex to relevant
  // information that lives on its baseline patch, i.e. all of the cells
  // directly attached to it.
  //
  // These data structures enable working from a nodal perspective rather than
  // an elemental perspective. They can be used to perform loops over data
  // defined uniquely at each node of a patch, rather than having to loop over
  // the elements in the patch and extract this data element by element.
  //
  // This function assumes that the bases you are working with are all nodal
  // bases.
  //
  // `vertex_to_vertex` is a vector of sets, with each set containing the
  // vertices that live on the baseline patch of the vertex that set is
  // associated with.
  //
  // `vertex_to_dof` is a vector of sets, with each set containing the
  // DOFs that live on the baseline patch of the vertex that set is
  // associated with.
  //
  // `vertex_to_dof` is a vector of vectors, with each vector containing the
  // enriched DOFs that live on the baseline patch of the vertex that set is
  // associated with. The reason this is a vector of vectors and not a vector
  // of sets is that order matters here, since the order needs to match the
  // order of the entries of `vertex_to_weight`.
  template <int dim>
  void create_vertex_mappings(const DoFHandler<dim>& dof_handler,
                              const DoFHandler<dim>& dof_handler_enriched,
                              std::vector<std::set<types::global_vertex_index>>& vertex_to_vertex,
                              std::vector<std::set<types::global_dof_index>>& vertex_to_dof,
                              std::vector<std::vector<types::global_dof_index>>& vertex_to_dof_enriched,
                              std::vector<std::vector<double>>& vertex_to_weight)
  {
    // Base finite element field
    const FiniteElement<dim>& fe = dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const Triangulation<dim>& triangulation = dof_handler.get_triangulation();

    // Enriched finite element field
    const FiniteElement<dim>& fe_enriched = dof_handler_enriched.get_fe();
    const unsigned int dofs_per_cell_enriched = fe_enriched.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices_enriched(dofs_per_cell_enriched);

    // Reference coordinates of the enriched Lagrange nodes
    Quadrature<dim> lagrange_nodes_enriched(fe_enriched.get_unit_support_points());

    // P1 finite element used to compute barycentric coordinates
    const FE_SimplexP<dim> fe_barycentric(1);
    // Q1 mapping used for barycentric coordinates
    MappingP1<dim> mapping_barycentric;
    // DoFHandler used for barycentric coordinates. We create this because deal.ii
    // doesn't like an FEValues object to be "reinit"ed using a DoFHandler that
    // was created using a different finite element than what is passed to the
    // FEValues object. That mismatch triggers an Assert only in Debug mode; in
    // Release mode, it is allowed.
    DoFHandler<dim> dof_handler_barycentric(triangulation);
    dof_handler_barycentric.distribute_dofs(fe_barycentric);

    // Used to get barycentric coordinates of the enriched Lagrange nodes
    FEValues<dim> barycentric_nodes_enriched(mapping_barycentric,
                                             fe_barycentric,
                                             lagrange_nodes_enriched,
                                             update_values);

    // Define size for all vectors of mappings
    vertex_to_vertex.resize(triangulation.n_vertices());
    vertex_to_dof.resize(triangulation.n_vertices());
    vertex_to_dof_enriched.resize(triangulation.n_vertices());
    vertex_to_weight.resize(triangulation.n_vertices());

    // Get iterator for enriched field explicitly. We need to take care of incrementing
    // this iterator manually, so that it keeps up with the iterator we're looping over.
    radial::cell_pointer<dim> cell_enriched = dof_handler_enriched.begin();

    radial::cell_pointer<dim> cell_barycentric = dof_handler_barycentric.begin();

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // Get information about DOFs and barycentric weights on this cell
      cell->get_dof_indices(local_dof_indices);
      cell_enriched->get_dof_indices(local_dof_indices_enriched);
      barycentric_nodes_enriched.reinit(cell_barycentric);

      for (const auto v_patch : cell->vertex_indices())
      {
        // The patch-central vertex whose patch we are adding to
        types::global_vertex_index vertex = cell->vertex_index(v_patch);

        // Add this cell's vertices to the patch
        for (const auto v_cell : cell->vertex_indices())
        {
          types::global_vertex_index neighbor = cell->vertex_index(v_cell);
          vertex_to_vertex[vertex].insert(neighbor);
        }

        // Add this cell's solution DOFs to the patch
        vertex_to_dof[vertex].insert(local_dof_indices.begin(),
                                     local_dof_indices.end());

        // Add this cell's enriched DOFs and barycentric weights to the patch
        for (unsigned int i = 0; i < dofs_per_cell_enriched; i++)
        {
          types::global_dof_index dof = local_dof_indices_enriched[i];

          bool dof_exists = (std::find(vertex_to_dof_enriched[vertex].begin(),
                                       vertex_to_dof_enriched[vertex].end(),
                                       dof) != vertex_to_dof_enriched[vertex].end());
          if (!dof_exists)
          {
            vertex_to_dof_enriched[vertex].push_back(dof);

            double weight = barycentric_nodes_enriched.shape_value(v_patch, i);
            vertex_to_weight[vertex].push_back(weight);
          }
        }
      }

      // These iterators need to be incrememnted manually
      ++cell_enriched;
      ++cell_barycentric;
    }
  }

  // Create a set of basis functions representing a global polynomial over a
  // patch of elements, to be used for a least-squares problem on the patch.
  template <int dim>
  void create_patch_basis(const unsigned int order,
                          radial::patch_basis<dim>& patch_basis_funcs)
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
  // Assumes straight-sided elements, so should not be used with curved meshes.
  template <int dim>
  void find_patch_bounding_box(const DoFHandler<dim>& dof_handler,
                               const std::set<types::global_vertex_index>& patch_vertices,
                               Point<dim>& coord_min, Point<dim>& coord_max)
  {
    const Triangulation<dim>& triangulation = dof_handler.get_triangulation();
    const std::vector<Point<dim>>& vertex_coords = triangulation.get_vertices();

    std::vector<std::vector<double>> coord_patch_vertices(dim);
    for (int d = 0; d < dim; d++)
      coord_patch_vertices[d].resize(patch_vertices.size());

    int vertex_count = 0;
    for (const auto& vertex : patch_vertices)
    {
      for (int d = 0; d < dim; d++)
        coord_patch_vertices[d][vertex_count] = vertex_coords[vertex](d);

      vertex_count++;
    }

    // Find limits of the bounding box that contains the patch
    for (int d = 0; d < dim; d++)
    {
      coord_min(d) = *std::min_element(coord_patch_vertices[d].begin(),
                                       coord_patch_vertices[d].end());
      coord_max(d) = *std::max_element(coord_patch_vertices[d].begin(),
                                       coord_patch_vertices[d].end());
    }
  }

  // Find the bounding box of a patch of cells.
  //
  // Implemented by finding the minimum and maximum physical coordinates over
  // all nodes in the patch. Note that we use nodes rather than vertices so that
  // this computation will be valid for curved meshes as well as linear meshes.
  // On a curved mesh, it's possible that the minimum/maximum coordinates will
  // come from an edge node rather than a vertex.
  //
  // The vector `dof_coords` is expected to contain the physical coordinates of
  // all of the nodes in the mesh.
  template <int dim>
  void find_patch_bounding_box(const std::vector<Point<dim>>& dof_coords,
                               const std::set<types::global_dof_index>& patch_dofs,
                               Point<dim>& coord_min, Point<dim>& coord_max)
  {
    std::vector<std::vector<double>> coord_patch_nodes(dim);
    for (int d = 0; d < dim; d++)
      coord_patch_nodes[d].resize(patch_dofs.size());

    int node_count = 0;
    for (const auto& dof : patch_dofs)
    {
      for (int d = 0; d < dim; d++)
        coord_patch_nodes[d][node_count] = dof_coords[dof](d);

      node_count++;
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

  // Find the bounding box of a patch of cells.
  //
  // Implemented by finding the minimum and maximum physical coordinates over
  // all nodes in the patch. Note that we use nodes rather than vertices so that
  // this computation will be valid for curved meshes as well as linear meshes.
  // On a curved mesh, it's possible that the minimum/maximum coordinates will
  // come from an edge node rather than a vertex.
  //
  // In this version of the function, the nodal coordinates are obtained by
  // evaluating them on the fly on each element, rather than extracting them
  // from a global vector of pre-computed nodal coordinates.
  template <int dim>
  void find_patch_bounding_box(const std::vector<radial::cell_pointer<dim>>& patch_cells,
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

          traversed_nodes.insert(local_dof_indices[i]);
        }
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

  // Construct a least-squares problem on a patch of cells, and solve it if the
  // system matrix is well-conditioned enough.
  //
  // The inputs can be roughly grouped into 4 groups as follows:
  // 1. `patch_cells`, `patch_dofs`, `patch_coord_min`, and `patch_coord_max`
  //    provide the details that identify this patch, i.e. the cells and DOFs
  //    contained in the patch as well as the bounding box of the patch. These
  //    are purely input arguments and do not need to be modified.
  // 2. `patch_basis_funcs` and `fe` provide the basis functions for the patch
  //    polyomial whose coefficients are to be determined by least squares and
  //    for a single finite element, respectively. These are also purely input
  //    arguments.
  // 3. `solution` is (a reference to) the Vector with the DOFs of the finite
  //    element field we are recovering from. `fe_values_nodes` provides the
  //    mechanism for extracting the particular values from `solution` that
  //    we need when we are on a cell in the patch. `fe_values_nodes` is not
  //    `const`, but it's not really an output, either. It's really only used
  //    internally and we don't need it once we're outside this function,
  //    but re-defining it for each patch seems wasteful since at the time
  //    of its construction it contains general information that is valid
  //    for any patch.
  // 4. `lsq_coeffs` and `rcond` are the coefficients obtained from the
  //    least-squares solve and the estimated reciprocal condition number of the
  //    least-squares system, respectively. A nuance about `lsq_coeffs` is that
  //    it is only actually modified if the least-squares system is considered
  //    to be well-conditioned enough to solve. Otherwise, the solve is skipped,
  //    and `lsq_coeffs` passes through this function without being modified.
  //    On the other hand, `rcond` will always have a value after this function,
  //    because estimating it does not require the solve step to actually happen
  //    since it can be estimated purely based on the QR decompisition.
  template<int dim>
  void least_squares_patch(const std::vector<radial::cell_pointer<dim>>& patch_cells,
                           const std::set<types::global_dof_index>& patch_dofs,
                           const Point<dim>& patch_coord_min,
                           const Point<dim>& patch_coord_max,
                           const radial::patch_basis<dim>& patch_basis_funcs,
                           const FiniteElement<dim>& fe,
                           const Vector<double>& solution,
                           FEValues<dim>& fe_values_nodes,
                           Vector<double>& lsq_coeffs,
                           double& rcond)
  {
    const unsigned int order_enriched = fe.degree + 1;
    unsigned int min_points = radial::get_min_points<dim>(order_enriched);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Vector to store values at the Lagrange nodes of an element
    std::vector<double> solution_values(fe_values_nodes.n_quadrature_points);

    // Create RHS and system matrix for discrete least-squares. We use GSL
    // so that condition number estimation can be done once the system
    // matrix is filled.
    gsl_vector *rhs = gsl_vector_alloc(patch_dofs.size());
    gsl_matrix *A = gsl_matrix_alloc(patch_dofs.size(), min_points);

    std::set<types::global_dof_index> eval_dofs;
    unsigned int eval_count = 0;

    for (const auto &cell: patch_cells)
    {
      fe_values_nodes.reinit(cell);

      cell->get_dof_indices(local_dof_indices);

      // Get values of the finite element field at the Lagrange nodes
      fe_values_nodes.get_function_values(solution, solution_values);
      
      for (const unsigned int i : fe_values_nodes.quadrature_point_indices())
      {
        if (eval_dofs.count(local_dof_indices[i]) < 1) // if no one has sampled at this node yet
        {
          // Sample solution at the patch node
          gsl_vector_set(rhs, eval_count, solution_values[i]);

          Point<dim> node_physical_coords = fe_values_nodes.quadrature_point(i);

          Point<dim> node_scaled_coords;
          for (int d = 0; d < dim; d++)
          {
            node_scaled_coords(d) = -1.0 +
                                    2.0*(node_physical_coords(d) - patch_coord_min(d)) /
                                    (patch_coord_max(d) - patch_coord_min(d));
          }

          for (unsigned int monomial_index = 0; monomial_index < min_points; monomial_index++)
          {
            gsl_matrix_set(A, eval_count, monomial_index,
                           patch_basis_funcs[monomial_index](node_scaled_coords));
          }

          eval_count++;

          eval_dofs.insert(local_dof_indices[i]);
        }
      }
    }

    // Compute QR decomposition of least-squares system matrix
    gsl_vector *tau = gsl_vector_alloc(min_points);
    gsl_linalg_QR_decomp(A, tau);

    // Estimate reciprocal condition number
    gsl_vector *work = gsl_vector_alloc(3 * min_points);
    gsl_linalg_QR_rcond(A, &rcond, work);
    gsl_vector_free(work);

    // If the condition number is good enough, solve the system
    double rcond_tol = std::numeric_limits<double>::epsilon() * 1e1;
    if (rcond > rcond_tol)
    {
      gsl_vector *x = gsl_vector_alloc(min_points);
      gsl_vector *residual = gsl_vector_alloc(patch_dofs.size());

      gsl_linalg_QR_lssolve(A, tau, rhs, x, residual);

      // Copy solution into deal.ii Vector
      for (unsigned int i = 0; i < min_points; i++)
        lsq_coeffs(i) = gsl_vector_get(x, i);

      gsl_vector_free(x);
      gsl_vector_free(residual);
    }

    gsl_matrix_free(A);
    gsl_vector_free(tau);
    gsl_vector_free(rhs);
  }
} // namespace radial