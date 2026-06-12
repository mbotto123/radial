#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <set>
#include <functional>

#include <recovery_common.h>

namespace radial
{
  using namespace dealii;

  template <int dim>
  void recover_solution_ppr(const DoFHandler<dim>& dof_handler, const MappingP1<dim>& mapping,
                            const Vector<double>& solution,
                            const DoFHandler<dim>& dof_handler_enriched,
                            Vector<double>& solution_enriched)
  {
    // TODO: There should be a check for whether or not the mesh contains any
    // curved elements. If it does, then MappingFE should be used. Currently,
    // since MappingP1 is hardcoded, this function assumes straight-sided
    // elements.

    //-------------------------------------------------------------------------//
    // Base finite element field
    const FiniteElement<dim>& fe = dof_handler.get_fe();
    const unsigned int order = fe.degree;

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Reference coordinates of the Lagrange nodes
    // (also used to get the values of the base finite element field at those nodes)
    Quadrature<dim> lagrange_nodes(fe.get_unit_support_points());
    FEValues<dim> fe_values_nodes(mapping,
                                  fe,
                                  lagrange_nodes,
                                  update_values | update_quadrature_points);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Enriched finite element field
    const unsigned int order_enriched = order + 1;
    const FiniteElement<dim>& fe_enriched = dof_handler_enriched.get_fe();

    const unsigned int dofs_per_cell_enriched = fe_enriched.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices_enriched(dofs_per_cell_enriched);

    // Reference coordinates of the Lagrange nodes
    Quadrature<dim> lagrange_nodes_enriched(fe_enriched.get_unit_support_points());
    FEValues<dim> fe_values_nodes_enriched(mapping,
                                           fe_enriched,
                                           lagrange_nodes_enriched,
                                           update_quadrature_points); // don't need values
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Create data structure that contains the baseline patch for every vertex.

    std::vector<std::vector<radial::cell_pointer<dim>>> vertex_to_cell;
    std::vector<std::vector<radial::cell_pointer<dim>>> vertex_to_cell_enriched;

    radial::create_vertex_to_cell(dof_handler, dof_handler_enriched,
                                  vertex_to_cell, vertex_to_cell_enriched);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Create monomial basis for least-squares fit

    unsigned int min_points = radial::get_min_points<dim>(order_enriched);

    std::vector<std::function<double(Point<dim>)>> patch_basis_funcs(min_points);
    radial::create_patch_basis(order, patch_basis_funcs);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Loop through vertices to construct recovery patches

    for (unsigned int v = 0; v < vertex_to_cell.size(); v++)
    {
      // Pointers to all cells in the patch. Initially, this is the same as the
      // baseline patch, but we may need to grow the patch beyond that.
      std::vector<radial::cell_pointer<dim>> patch_cells = vertex_to_cell[v];
      // Indices of all cells in the patch
      std::set<int> patch_cell_indices;
      // Global DOF indices of all DOFs in the patch
      std::set<types::global_dof_index> patch_dofs;
      // Global vertex indices of all vertices in the patch
      std::set<unsigned int> patch_vertices;
      // Global vertex indices of the vertices on the outer patch boundary
      std::set<unsigned int> neighbors;

      // Initialize sets based on the baseline patch
      for (const auto &cell: vertex_to_cell[v])
      {
        patch_cell_indices.insert(cell->index());

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i : fe_values_nodes.dof_indices())
          patch_dofs.insert(local_dof_indices[i]);

        for (const auto v_patch: cell->vertex_indices())
        {
          unsigned int neighbor = cell->vertex_index(v_patch);

          patch_vertices.insert(neighbor);

          if (neighbor != v)
            neighbors.insert(neighbor);
        }
      }

      // Vector of least-squares coefficients
      Vector<double> a(min_points);
      // Reciprocal condition number of the least-squares system on the patch
      double rcond;
      // The reciprocal condition number value at which we consider the
      // least-squares system to be too ill-conditioned to attempt solving.
      double rcond_tol = std::numeric_limits<double>::epsilon() * 1e1;

      // Try least-squares on baseline patch if it already has enough points to
      // create a solvable least-squares system. If this is successful, i.e. if
      // the system is well-conditioned enough, then we don't need to grow the
      // patch at all.
      if (patch_dofs.size() > min_points)
      {
        radial::least_squares_patch(patch_cells, patch_dofs,
                                    patch_basis_funcs, fe,
                                    solution, fe_values_nodes,
                                    a, rcond);
      }
      else
      {
        // If we don't have enough points to do least-squares yet, just set
        // the reciprocal condition number to zero to indicate that we don't
        // have a solvable system yet. There's no way to estimate it since we
        // can't even construct a least-squares system yet.
        rcond = 0;
      }

      int growth_iter = 0;
      const int max_iter = 3;

      // TODO: Optionally add an additional check based on the size of the
      // least-squares residual norm? User could provide an acceptable tolerance.
      while ((growth_iter < max_iter) &&                /* max iters exceeded */
             (patch_dofs.size() <= min_points ||        /* not enough points  */
             (rcond < rcond_tol || std::isnan(rcond)))) /* ill-conditioning   */
      {
        // Grow by one layer by adding all cells that contain vertices that lie on patch boundary
        for (const auto& neighbor : neighbors)
        {
          for (const auto &cell: vertex_to_cell[neighbor])
          {
            if (patch_cell_indices.count(cell->index()) < 1)
              patch_cells.push_back(cell);

            patch_cell_indices.insert(cell->index());

            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i : fe_values_nodes.dof_indices())
              patch_dofs.insert(local_dof_indices[i]);
          }
        }

        std::set<unsigned int> next_neighbors;

        // Determine indices of sampling points on new patch boundary defined by this growth iteration
        for (const auto& neighbor : neighbors)
        {
          std::set<unsigned int> neighbors_of_neighbor;

          for (const auto &cell: vertex_to_cell[neighbor])
          {
            for (const auto v_patch: cell->vertex_indices())
            {
              unsigned int neighbor_of_neighbor = cell->vertex_index(v_patch);

              if (neighbor_of_neighbor != neighbor)
                neighbors_of_neighbor.insert(neighbor_of_neighbor);
            }
          }

          for (const auto& neighbor_of_neighbor : neighbors_of_neighbor)
          {
            if (patch_vertices.count(neighbor_of_neighbor) < 1)
              next_neighbors.insert(neighbor_of_neighbor);
          }
        }

        // Update overall list of patch vertices
        for (const auto& next_neighbor : next_neighbors)
          patch_vertices.insert(next_neighbor);

        neighbors = next_neighbors;

        // Try least-squares if we have enough points. If the system is too
        // ill-conditioned, the solve step will be skipped, and the rcond value
        // will indicate that we should keep growing.
        if (patch_dofs.size() > min_points)
        {
          radial::least_squares_patch(patch_cells, patch_dofs,
                                      patch_basis_funcs, fe,
                                      solution, fe_values_nodes,
                                      a, rcond);
        }

        growth_iter++;
      }

      // Fail if max patch growth iteration was not enough to satisfy the
      // conditions for a solvable and acceptably-conditioned system.
      Assert(patch_dofs.size() >= min_points + 1,
             ExcMessage("Recovery patch doesn't have enough sampling points!"));
      Assert((rcond > rcond_tol) && !std::isnan(rcond),
             ExcMessage("Least-squares system is too ill-conditioned to solve!"));

      // Points to store coordinates of patch bounding box
      Point<dim> coord_min, coord_max;
      radial::find_patch_bounding_box(patch_cells, patch_dofs,
                                      fe_values_nodes, local_dof_indices,
                                      coord_min, coord_max);

      // Evaluate recovered solution polynomials at selected locations on patch
      // (nodes that are interior to edges attached to the patch-central vertex,
      // and cell nodes of cells that contain the patch-central vertex)

      std::set<types::global_dof_index> eval_dofs_enriched;

      for (const auto &cell : vertex_to_cell_enriched[v])
      {
        fe_values_nodes_enriched.reinit(cell);

        cell->get_dof_indices(local_dof_indices_enriched);

        // Find the local vertex index of the patch-central vertex
        unsigned int central_vert_local_index;
        bool central_vert_found = false;
        for (const auto v_enriched : cell->vertex_indices())
        {
          if (cell->vertex_index(v_enriched) == v)
          {
            central_vert_local_index = v_enriched;
            central_vert_found = true;
          }
        }
        
        Assert(central_vert_found,
               ExcMessage("Central vertex of recovery patch not found!"));

        // Evaluate recovered solution at edge and cell nodes
        for (const unsigned int i : fe_values_nodes_enriched.quadrature_point_indices())
        {
          // Node coordinates in reference space
          Point<dim> node_ref_coords = fe_enriched.unit_support_point(i);

          std::vector<double> node_ref_barycentric(dim + 1);
          node_ref_barycentric[0] = 1.0;
          for (int d = 0; d < dim; d++)
          {
            node_ref_barycentric[0] -= node_ref_coords[d];
            node_ref_barycentric[d + 1] = node_ref_coords[d];
          }

          double node_ref_barycentric_patch = node_ref_barycentric[central_vert_local_index];

          Point<dim> node_physical_coords = fe_values_nodes_enriched.quadrature_point(i);

          Point<dim> node_scaled_coords;
          for (int d = 0; d < dim; d++)
            node_scaled_coords(d) = -1.0 + 2.0*(node_physical_coords(d) - coord_min(d))/(coord_max(d) - coord_min(d));

          double solution_enriched_node = 0;
          for (unsigned int monomial_index = 0; monomial_index < min_points; monomial_index++)
            solution_enriched_node += a(monomial_index) * patch_basis_funcs[monomial_index](node_scaled_coords);

          if (eval_dofs_enriched.count(local_dof_indices_enriched[i]) < 1)
            solution_enriched(local_dof_indices_enriched[i]) += node_ref_barycentric_patch * solution_enriched_node;

          eval_dofs_enriched.insert(local_dof_indices_enriched[i]);
        }
      }
    }
    //-------------------------------------------------------------------------//
  }
} // namespace radial
