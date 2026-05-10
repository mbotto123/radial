#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/householder.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <set>
#include <functional>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

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

    // Vector to store values at the Lagrange nodes of an element
    std::vector<double> solution_values(lagrange_nodes.size());
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
    //
    // The way a "patch" is implemented is as a list of iterators, i.e. pointers
    // to the different cells in the patch. These pointers may be pointing to
    // cells that are far away from each other in memory, so maybe there is a
    // more efficient way to implement this in terms of memory access. TODO.
    //
    // Two data structures are created, one for the non-enriched field, and one
    // for the enriched field.

    const Triangulation<dim>& triangulation = dof_handler.get_triangulation();

    std::vector<std::list<typename DoFHandler<dim>::active_cell_iterator>> vertex_to_cell;
    vertex_to_cell.resize(triangulation.n_vertices());

    std::vector<std::list<typename DoFHandler<dim>::active_cell_iterator>> vertex_to_cell_enriched;
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
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Create monomial basis for least-squares fit

    // Minimum number of sampling points required to get a solvable system on a patch
    unsigned int min_points;
    if (dim == 2)
    {
      min_points = 0.5 * (order_enriched + 1) * (order_enriched + 2);
    }
    else if (dim == 3)
    {
      min_points = (1.0/6.0) * (order_enriched + 1) * (order_enriched + 2) * (order_enriched + 3);
    }

    std::vector<std::function<double(Point<dim>)>> patch_basis_funcs(min_points);

    radial::create_patch_basis(order, patch_basis_funcs);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Loop through vertices to construct recovery patches

    unsigned int min_points_linear;
    if (dim == 2)
    {
      min_points_linear = 0.5 * (2 + 1) * (2 + 2);
    }
    else if (dim == 3)
    {
      min_points_linear = (1.0/6.0) * (2 + 1) * (2 + 2) * (2 + 3);
    }

    const std::vector<Point<dim>> &vertex_coords = triangulation.get_vertices();

    for (unsigned int v = 0; v < vertex_to_cell.size(); v++)
    {
      std::set<typename DoFHandler<dim>::active_cell_iterator> patch_cells;

      // Set to keep count of patch DOFs
      std::set<types::global_dof_index> patch_dofs;

      // Add cells that contain the central vertex to the patch
      for (const auto &cell: vertex_to_cell[v])
      {
        patch_cells.insert(cell);
   
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i : fe_values_nodes.dof_indices())
          patch_dofs.insert(local_dof_indices[i]);
      }

      std::set<unsigned int> patch_vertices;
      std::set<unsigned int> neighbors;

      for (const auto &cell: vertex_to_cell[v])
      {
        for (const auto v: cell->vertex_indices())
        {
          unsigned int neighbor = cell->vertex_index(v);

          patch_vertices.insert(neighbor);

          if (neighbor != v)
            neighbors.insert(neighbor);
        }
      }

      unsigned int nverts = patch_vertices.size();

      // Vector of least-squares coefficients
      Vector<double> a(min_points);

      // Points to store coordinates of patch bounding box
      Point<dim> coord_min, coord_max;

      // Reciprocal condition number of the least-squares system on the patch
      double rcond;

      // The reciprocal condition number value at which we consider the
      // least-squares system to be too ill-conditioned to attempt solving.
      double rcond_tol = std::numeric_limits<double>::epsilon();

      // Try least-squares on baseline patch if it already has enough points to
      // create a solvable least-squares system. If this is successful, i.e. if
      // the system is well-conditioned enough, then we don't need to grow the
      // patch at all.
      if (patch_dofs.size() > min_points)
      {
        std::vector<std::vector<double>> coord_patch_vertices(dim);
        for (int d = 0; d < dim; d++)
          coord_patch_vertices[d].resize(nverts);

        int vertex_count = 0;
        for (const auto& vertex : patch_vertices)
        {
          for (int d = 0; d < dim; d++)
            coord_patch_vertices[d][vertex_count] = vertex_coords[vertex](d);

          vertex_count++;
        }

        // Find limits of the bounding box that contains the patch
        // TODO: For this to be valid on curved elements, this needs to be done
        // with node coordinates, not vertex coordinates.
        for (int d = 0; d < dim; d++)
        {
          coord_min(d) = *std::min_element(coord_patch_vertices[d].begin(), coord_patch_vertices[d].end());
          coord_max(d) = *std::max_element(coord_patch_vertices[d].begin(), coord_patch_vertices[d].end());
        }

        // Create RHS and system matrix for discrete least-squares. We use GSL
        // so that condition number estimation can be done once the system
        // matrix is filled.
        gsl_vector *rhs = gsl_vector_alloc(patch_dofs.size());
        gsl_matrix *A = gsl_matrix_alloc(patch_dofs.size(), min_points);

        // Discrete least-squares

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
                node_scaled_coords(d) = -1.0 + 2.0*(node_physical_coords(d) - coord_min(d))/(coord_max(d) - coord_min(d));

              for (unsigned int monomial_index = 0; monomial_index < min_points; monomial_index++)
              {
                gsl_matrix_set(A, eval_count, monomial_index,
                               patch_basis_funcs[monomial_index](node_scaled_coords));
              }

              eval_count++;
            }
            eval_dofs.insert(local_dof_indices[i]);
          }
        }

        // Compute QR decomposition of least-squares system matrix
        gsl_matrix *T = gsl_matrix_alloc(min_points, min_points);
        gsl_linalg_QR_decomp_r(A, T);

        // Estimate reciprocal condition number
        gsl_vector *work = gsl_vector_alloc(3 * min_points);
        gsl_linalg_QR_rcond(A, &rcond, work);
        gsl_vector_free(work);

        // If the condition number is good enough, solve the system
        if (rcond > rcond_tol)
        {
          // The solution only actually has size N, but GSL asks for
          // this input to have size M. The entries beyond N-1 store a vector
          // that can be used to compute the least-squares residual norm.
          gsl_vector *x = gsl_vector_alloc(patch_dofs.size());

          gsl_vector *work = gsl_vector_alloc(min_points);
          gsl_linalg_QR_lssolve_r(A, T, rhs, x, work);
          gsl_vector_free(work);

          // Copy solution into deal.ii Vector
          for (unsigned int i = 0; i < min_points; i++)
            a(i) = gsl_vector_get(x, i);

          gsl_vector_free(x);
        }

        gsl_matrix_free(A);
        gsl_matrix_free(T);
        gsl_vector_free(rhs);
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
      while ((growth_iter < max_iter) &&
             (patch_dofs.size() <= min_points || rcond < rcond_tol))
      {
        // Grow by one layer by adding all cells that contain vertices that lie on patch boundary
        for (const auto& neighbor : neighbors)
        {
          for (const auto &cell: vertex_to_cell[neighbor])
          {
            patch_cells.insert(cell);

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
            for (const auto v: cell->vertex_indices())
            {
              unsigned int neighbor_of_neighbor = cell->vertex_index(v);

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

        nverts = patch_vertices.size();

        if (patch_dofs.size() > min_points)
        {
          // If we have enough points, try least-squares and check residual norm

          std::vector<std::vector<double>> coord_patch_vertices(dim);
          for (int d = 0; d < dim; d++)
            coord_patch_vertices[d].resize(nverts);

          int vertex_count = 0;
          for (const auto& vertex : patch_vertices)
          {
            for (int d = 0; d < dim; d++)
              coord_patch_vertices[d][vertex_count] = vertex_coords[vertex](d);

            vertex_count++;
          }

          // Find limits of the bounding box that contains the patch
          // TODO: For this to be valid on curved elements, this needs to be done
          // with node coordinates, not vertex coordinates.
          for (int d = 0; d < dim; d++)
          {
            coord_min(d) = *std::min_element(coord_patch_vertices[d].begin(), coord_patch_vertices[d].end());
            coord_max(d) = *std::max_element(coord_patch_vertices[d].begin(), coord_patch_vertices[d].end());
          }

          // Create RHS and system matrix for discrete least-squares. We use GSL
          // so that condition number estimation can be done once the system
          // matrix is filled.
          gsl_vector *rhs = gsl_vector_alloc(patch_dofs.size());
          gsl_matrix *A = gsl_matrix_alloc(patch_dofs.size(), min_points);

          // Discrete least-squares

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
                  node_scaled_coords(d) = -1.0 + 2.0*(node_physical_coords(d) - coord_min(d))/(coord_max(d) - coord_min(d));

                for (unsigned int monomial_index = 0; monomial_index < min_points; monomial_index++)
                {
                  gsl_matrix_set(A, eval_count, monomial_index,
                                patch_basis_funcs[monomial_index](node_scaled_coords));
                }
                eval_count++;
              }
              eval_dofs.insert(local_dof_indices[i]);
            }
          }

          // Compute QR decomposition of least-squares system matrix
          gsl_matrix *T = gsl_matrix_alloc(min_points, min_points);
          gsl_linalg_QR_decomp_r(A, T);

          // Estimate reciprocal condition number
          gsl_vector *work = gsl_vector_alloc(3 * min_points);
          gsl_linalg_QR_rcond(A, &rcond, work);
          gsl_vector_free(work);

          // If the condition number is good enough, solve the system
          if (rcond > rcond_tol)
          {
            // The solution only actually has size N, but GSL asks for
            // this input to have size M. The entries beyond N-1 store a vector
            // that can be used to compute the least-squares residual norm.
            gsl_vector *x = gsl_vector_alloc(patch_dofs.size());

            gsl_vector *work = gsl_vector_alloc(min_points);
            gsl_linalg_QR_lssolve_r(A, T, rhs, x, work);
            gsl_vector_free(work);

            // Copy solution into deal.ii Vector
            for (unsigned int i = 0; i < min_points; i++)
              a(i) = gsl_vector_get(x, i);

            gsl_vector_free(x);
          }

          gsl_matrix_free(A);
          gsl_matrix_free(T);
          gsl_vector_free(rhs);
        }

        growth_iter++;
      }

      // Fail if max patch growth iteration was not enough to satisfy the
      // conditions for a solvable and acceptably-conditioned system.
      Assert(patch_dofs.size() >= min_points + 1,
             ExcMessage("Recovery patch doesn't have enough sampling points!"));
      Assert(rcond > rcond_tol,
             ExcMessage("Least-squares system is too ill-conditioned to solve!"));

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
