#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/point.h>
#include <deal.II/base/table.h>

// Uncomment if adding Legendre basis option later
// #include <gsl/gsl_sf_legendre.h>

#include <recovery_common.h>

// Functions common to both discrete and integral least-squares recovery

namespace radial
{
  using namespace dealii;

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

  // Given the coefficients of a polynomial obtained by solving a least-squares
  // problem on a patch, evaluate the polynomial at the nodes of the baseline
  // patch corresponding to the nodes of the enriched solution we want to compute.
  //
  // Note that we are not evaluating on every node of the patch that was actually
  // used to solve the least-squares problem. Rather, this is only the baseline
  // patch, i.e. all of the cells that are directly attached to a vertex.
  //
  // The `patch_weights` determine how much the value computed at each node
  // contributes to the enriched solution. The weights corresponding to the nodes
  // on the boundary of the baseline patch are zero, so those nodes don't end up
  // contributing anything.
  //
  // The inputs can be roughly grouped into 4 groups as follows:
  // 1. `dof_coords_enriched` is a "global" vector, i.e. it contains info
  //    defined over the entire mesh. It has the physical coordinates of all
  //    enriched solution nodes in the mesh.
  // 2. `patch_dofs_enriched` is a vector containing the global DOF indices of the
  //    enriched solution DOFs on the patch. These indices will be used to index
  //    into the global vector discussed above. `patch_weights` is a vector
  //    containing the barycentric weights corresponding to each global DOF in
  //    `patch_dofs_enriched`.
  // 3. `patch_coord_min`, `patch_coord_max`, `patch_basis_funcs`, and `lsq_coeffs`
  //    provide information necessary to evaluate the least-squares polynomial at the
  //    nodes of the patch. This information includes the coordinates of the
  //    bounding box that contains the patch (used to scale the coordinates of the
  //    patch nodes) as well as the patch basis functions and the polynomial
  //    coefficients.
  // 4. `solution_enriched` is another "global" vector. This is the vector into
  //    which the patch polynomial evaluations are accumulated to produce the
  //    nodal values of the enriched solution. This vector must be initialized
  //    with zeros, since we need to accumulate into it.
  template<int dim>
  void evaluate_patch_polynomial(const std::vector<Point<dim>>& dof_coords_enriched,
                                 const std::vector<types::global_dof_index>& patch_dofs_enriched,
                                 const std::vector<double>& patch_weights,
                                 const Point<dim>& patch_coord_min,
                                 const Point<dim>& patch_coord_max,
                                 const radial::patch_basis<dim>& patch_basis_funcs,
                                 const Vector<double>& lsq_coeffs,
                                 Vector<double>& solution_enriched)
  {
    int n_patch_dofs = patch_dofs_enriched.size();

    for (int patch_dof = 0; patch_dof < n_patch_dofs; patch_dof++)
    {
      types::global_dof_index dof = patch_dofs_enriched[patch_dof];

      Point<dim> node_physical_coords = dof_coords_enriched[dof];

      Point<dim> node_scaled_coords;
      for (int d = 0; d < dim; d++)
      {
        node_scaled_coords(d) = -1.0 +
                                2.0*(node_physical_coords(d) - patch_coord_min(d)) /
                                (patch_coord_max(d) - patch_coord_min(d));
      }

      double solution_enriched_node = 0;
      for (unsigned int monomial_index = 0; monomial_index < lsq_coeffs.size(); monomial_index++)
      {
        solution_enriched_node += lsq_coeffs(monomial_index) *
                                  patch_basis_funcs[monomial_index](node_scaled_coords);
      }

      // Accumulate this patch's enriched solution contribution, weighted by
      // the node's barycentric coordinate.
      solution_enriched(dof) += patch_weights[patch_dof] * solution_enriched_node;
    }
  }
} // namespace radial