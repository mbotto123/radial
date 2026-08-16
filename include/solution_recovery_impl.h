#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

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
  void recover_solution_ppr_discrete(const DoFHandler<dim>& dof_handler,
                                     const MappingP1<dim>& mapping,
                                     const Vector<double>& solution,
                                     const DoFHandler<dim>& dof_handler_enriched,
                                     Vector<double>& solution_enriched)
  {
    const FiniteElement<dim>& fe = dof_handler.get_fe();
    const unsigned int order = fe.degree;

    const FiniteElement<dim>& fe_enriched = dof_handler_enriched.get_fe();
    const unsigned int order_enriched = fe_enriched.degree;

    //------------------------------------------------------------------------//
    // Get physical coordinates of the nodes of the finite element solution and
    // enriched solution.

    std::vector<Point<dim>> dof_coords(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, dof_coords);

    std::vector<Point<dim>> dof_coords_enriched(dof_handler_enriched.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dof_handler_enriched,
                                         dof_coords_enriched);
    //------------------------------------------------------------------------//

    //------------------------------------------------------------------------//
    // Construct mapping information from vertices to information on their
    // baseline patch.

    std::vector<std::set<types::global_vertex_index>> vertex_to_vertex;
    std::vector<std::set<types::global_dof_index>> vertex_to_dof;
    std::vector<std::vector<types::global_dof_index>> vertex_to_dof_enriched;
    std::vector<std::vector<double>> vertex_to_weight;

    radial::create_vertex_mappings(dof_handler, dof_handler_enriched,
                                   vertex_to_vertex, vertex_to_dof,
                                   vertex_to_dof_enriched, vertex_to_weight);
    //------------------------------------------------------------------------//

    //------------------------------------------------------------------------//
    // Create monomial basis for least-squares fit

    unsigned int min_points = radial::get_min_points<dim>(order_enriched);

    radial::patch_basis<dim> patch_basis_funcs(min_points);
    radial::create_patch_basis(order, patch_basis_funcs);
    //------------------------------------------------------------------------//

    //------------------------------------------------------------------------//
    // Loop through vertices to construct recovery patches

    for (types::global_vertex_index v = 0; v < vertex_to_vertex.size(); v++)
    {
      // Global DOF indices of all DOFs in the patch
      std::set<types::global_dof_index> patch_dofs = vertex_to_dof[v];

      // Global vertex indices of all vertices in the patch
      std::set<types::global_vertex_index> patch_vertices = vertex_to_vertex[v];

      // Global vertex indices of the vertices on the patch boundary
      std::set<types::global_vertex_index> patch_boundary_vertices;

      // Before growing the patch, the patch boundary vertices are every vertex
      // except the patch-central one.
      patch_boundary_vertices = patch_vertices;
      patch_boundary_vertices.erase(v);

      // Vector of least-squares coefficients
      Vector<double> a(min_points);
      // Reciprocal condition number of the least-squares system on the patch
      double rcond;
      // The reciprocal condition number value at which we consider the
      // least-squares system to be too ill-conditioned to attempt solving.
      double rcond_tol = std::numeric_limits<double>::epsilon() * 1e1;

      // Points to store coordinates of patch bounding box
      Point<dim> coord_min, coord_max;

      // Try least-squares on baseline patch if it already has enough points to
      // create a solvable least-squares system. If this is successful, i.e. if
      // the system is well-conditioned enough, then we don't need to grow the
      // patch at all.
      if (patch_dofs.size() > min_points)
      {
        radial::find_patch_bounding_box(dof_coords, patch_dofs,
                                        coord_min, coord_max);
        radial::least_squares_patch_discrete(dof_coords, solution,
                                             patch_dofs, min_points,
                                             coord_min, coord_max,
                                             patch_basis_funcs,
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
        std::set<types::global_vertex_index> patch_vertices_old = patch_vertices;

        // Grow by one layer by looping through all of the patch boundary vertices
        // and adding all of their neighbor vertices and DOFs that aren't already
        // in the patch.
        for (const auto& v_boundary : patch_boundary_vertices)
        {
          patch_vertices.insert(vertex_to_vertex[v_boundary].begin(),
                                vertex_to_vertex[v_boundary].end());
          patch_dofs.insert(vertex_to_dof[v_boundary].begin(),
                            vertex_to_dof[v_boundary].end());
        }

        // Determine indices of vertices on new patch boundary defined by this
        // growth iteration.
        std::set<types::global_vertex_index> new_boundary;
        std::set_difference(patch_vertices.begin(), patch_vertices.end(),
                            patch_vertices_old.begin(), patch_vertices_old.end(),
                            std::inserter(new_boundary, new_boundary.begin()));
        patch_boundary_vertices = new_boundary;

        // Try least-squares if we have enough points. If the system is too
        // ill-conditioned, the solve step will be skipped, and the rcond value
        // will indicate that we should keep growing.
        if (patch_dofs.size() > min_points)
        {
          radial::find_patch_bounding_box(dof_coords, patch_dofs,
                                          coord_min, coord_max);
          radial::least_squares_patch_discrete(dof_coords, solution,
                                               patch_dofs, min_points,
                                               coord_min, coord_max,
                                               patch_basis_funcs,
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

      radial::evaluate_patch_polynomial(dof_coords_enriched,
                                        vertex_to_dof_enriched[v],
                                        vertex_to_weight[v],
                                        coord_min, coord_max,
                                        patch_basis_funcs, a, solution_enriched);
    }
    //------------------------------------------------------------------------//
  }

  template <int dim>
  void recover_solution_ppr_integral(const DoFHandler<dim>& dof_handler,
                                     const MappingP1<dim>& mapping,
                                     const Vector<double>& solution,
                                     const DoFHandler<dim>& dof_handler_enriched,
                                     Vector<double>& solution_enriched)
  {
    const FiniteElement<dim>& fe = dof_handler.get_fe();
    const unsigned int order = fe.degree;

    const FiniteElement<dim>& fe_enriched = dof_handler_enriched.get_fe();
    const unsigned int order_enriched = fe_enriched.degree;

    //------------------------------------------------------------------------//
    // Get physical coordinates of the nodes of the finite element solution and
    // enriched solution.

    std::vector<Point<dim>> dof_coords(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, dof_coords);

    std::vector<Point<dim>> dof_coords_enriched(dof_handler_enriched.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dof_handler_enriched,
                                         dof_coords_enriched);
    //------------------------------------------------------------------------//

    //------------------------------------------------------------------------//
    // Construct mapping information from vertices to information on their
    // baseline patch.

    std::vector<std::vector<radial::cell_pointer<dim>>> vertex_to_cell;
    std::vector<std::vector<radial::cell_pointer<dim>>> vertex_to_cell_enriched;

    // TODO: remove `vertex_to_cell_enriched`, which is no longer used
    radial::create_vertex_to_cell(dof_handler, dof_handler_enriched,
                                  vertex_to_cell, vertex_to_cell_enriched);

    std::vector<std::set<types::global_vertex_index>> vertex_to_vertex;
    std::vector<std::set<types::global_dof_index>> vertex_to_dof;
    std::vector<std::vector<types::global_dof_index>> vertex_to_dof_enriched;
    std::vector<std::vector<double>> vertex_to_weight;

    radial::create_vertex_mappings(dof_handler, dof_handler_enriched,
                                   vertex_to_vertex, vertex_to_dof,
                                   vertex_to_dof_enriched, vertex_to_weight);
    //------------------------------------------------------------------------//

    //------------------------------------------------------------------------//
    // Create monomial basis for least-squares fit

    unsigned int min_points = radial::get_min_points<dim>(order_enriched);

    radial::patch_basis<dim> patch_basis_funcs(min_points);
    radial::create_patch_basis(order, patch_basis_funcs);
    //------------------------------------------------------------------------//

    //------------------------------------------------------------------------//
    // Loop through vertices to construct recovery patches

    for (types::global_vertex_index v = 0; v < vertex_to_vertex.size(); v++)
    {
      // Pointers to all cells in the patch
      std::vector<radial::cell_pointer<dim>> patch_cells = vertex_to_cell[v];

      // Global cell indices of all cells in the patch
      std::set<types::global_cell_index> patch_cell_indices;

      // Global DOF indices of all DOFs in the patch
      std::set<types::global_dof_index> patch_dofs = vertex_to_dof[v];

      // Global vertex indices of all vertices in the patch
      std::set<types::global_vertex_index> patch_vertices = vertex_to_vertex[v];

      // Global vertex indices of the vertices on the patch boundary
      std::set<types::global_vertex_index> patch_boundary_vertices;

      for (const auto& cell : vertex_to_cell[v])
        patch_cell_indices.insert(cell->index());

      // Before growing the patch, the patch boundary vertices are every vertex
      // except the patch-central one.
      patch_boundary_vertices = patch_vertices;
      patch_boundary_vertices.erase(v);

      // Vector of least-squares coefficients
      Vector<double> a(min_points);

      int growth_iter = 0;
      const int max_iter = 3;

      // TODO: Optionally add an additional check based on the size of the
      // least-squares residual norm? User could provide an acceptable tolerance.
      while ((growth_iter < max_iter) &&        /* max iters exceeded */
             (patch_dofs.size() <= min_points)) /* not enough points  */
      {
        std::set<types::global_vertex_index> patch_vertices_old = patch_vertices;

        // Grow by one layer by looping through all of the patch boundary vertices
        // and adding all of their neighbor vertices and DOFs that aren't already
        // in the patch.
        for (const auto& v_boundary : patch_boundary_vertices)
        {
          for (const auto& cell : vertex_to_cell[v_boundary])
          {
            if (patch_cell_indices.count(cell->index()) < 1) /* cell is not in the patch yet */
            {
              patch_cells.push_back(cell);
              patch_cell_indices.insert(cell->index());
            }
          }

          patch_vertices.insert(vertex_to_vertex[v_boundary].begin(),
                                vertex_to_vertex[v_boundary].end());
          patch_dofs.insert(vertex_to_dof[v_boundary].begin(),
                            vertex_to_dof[v_boundary].end());
        }

        // Determine indices of vertices on new patch boundary defined by this
        // growth iteration.
        std::set<types::global_vertex_index> new_boundary;
        std::set_difference(patch_vertices.begin(), patch_vertices.end(),
                            patch_vertices_old.begin(), patch_vertices_old.end(),
                            std::inserter(new_boundary, new_boundary.begin()));
        patch_boundary_vertices = new_boundary;

        growth_iter++;
      }

      // Fail if max patch growth iteration was not enough to satisfy the
      // conditions for a solvable system.
      Assert(patch_dofs.size() >= min_points + 1,
             ExcMessage("Recovery patch doesn't have enough sampling points!"));

      Point<dim> coord_min, coord_max;
      radial::find_patch_bounding_box(dof_coords, patch_dofs,
                                      coord_min, coord_max);
      radial::least_squares_patch_integral(patch_cells, coord_min, coord_max,
                                           patch_basis_funcs, mapping, fe,
                                           solution, a);

      radial::evaluate_patch_polynomial(dof_coords_enriched,
                                        vertex_to_dof_enriched[v],
                                        vertex_to_weight[v],
                                        coord_min, coord_max,
                                        patch_basis_funcs, a, solution_enriched);
    }
    //------------------------------------------------------------------------//
  }

  template <int dim>
  void recover_solution_ppr(const DoFHandler<dim>& dof_handler, const MappingP1<dim>& mapping,
                            const Vector<double>& solution,
                            const DoFHandler<dim>& dof_handler_enriched,
                            Vector<double>& solution_enriched,
                            const bool use_integral_least_squares = false)
  {
    if (use_integral_least_squares)
    {
      recover_solution_ppr_integral(dof_handler, mapping, solution,
                                    dof_handler_enriched, solution_enriched);
    }
    else
    {
      recover_solution_ppr_discrete(dof_handler, mapping, solution,
                                    dof_handler_enriched, solution_enriched);
    }
  }
} // namespace radial
