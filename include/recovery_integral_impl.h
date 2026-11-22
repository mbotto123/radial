#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/lac/vector.h>

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/lapack_full_matrix.h>

#include <set>

#include <recovery_common.h>

// Functions specific to integral least-squares recovery, including the recovery
// function itself.

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
  template <int dim>
  void create_vertex_to_cell(const DoFHandler<dim>& dof_handler,
                             std::vector<std::vector<radial::cell_pointer<dim>>>& vertex_to_cell)
  {
    const Triangulation<dim>& triangulation = dof_handler.get_triangulation();

    vertex_to_cell.resize(triangulation.n_vertices());

    for (const auto &cell: dof_handler.active_cell_iterators())
    {
      for (const auto v: cell->vertex_indices())
      {
        // Add base field cell
        vertex_to_cell[cell->vertex_index(v)].push_back(cell);
      }
    }
  }

  template<int dim>
  void least_squares_patch_integral(const std::vector<radial::cell_pointer<dim>>& patch_cells,
                                    const Point<dim>& patch_coord_min,
                                    const Point<dim>& patch_coord_max,
                                    const radial::patch_basis<dim>& patch_basis_funcs,
                                    const MappingP1<dim>& mapping,
                                    const FiniteElement<dim>& fe,
                                    const Vector<double>& solution,
                                    Vector<double>& lsq_coeffs)
  {
    const unsigned int order_enriched = fe.degree + 1;
    unsigned int min_points = radial::get_min_points<dim>(order_enriched);

    // This quadrature formula is somewhat hardcoded to assume that the finite
    // element solution was obtained using a quadrature formula of p + 1. If
    // this assumption holds, then using p' + 1 here ensures that the integral
    // least-squares fit is computed with a higher quadrature order than the
    // finite element solution, which is what we want.
    const QGaussSimplex<dim> quadrature_formula(order_enriched + 1);

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> solution_values(quadrature_formula.size());

    Vector<double> rhs(min_points);
    LAPACKFullMatrix<double> A(min_points, min_points);

    rhs = 0;
    A = 0;

    for (const auto &cell: patch_cells)
    {
      fe_values.reinit(cell);
      fe_values.get_function_values(solution, solution_values);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        const Point<dim> &x_q = fe_values.quadrature_point(q_index);

        Point<dim> x_q_scaled;
        for (int d = 0; d < dim; d++)
        {
          x_q_scaled(d) = -1.0 + 2.0*(x_q(d) - patch_coord_min(d)) /
                          (patch_coord_max(d) - patch_coord_min(d));
        }

        for (unsigned int i = 0; i < min_points; i++)
        {
          for (unsigned int j = 0; j < min_points; j++)
          {
            A(i,j) += patch_basis_funcs[i](x_q_scaled) *
                      patch_basis_funcs[j](x_q_scaled) *
                      fe_values.JxW(q_index);
          }

          rhs(i) += patch_basis_funcs[i](x_q_scaled) *
                    solution_values[q_index] *
                    fe_values.JxW(q_index);
        }
      }
    }

    A.set_property(LAPACKSupport::Property::symmetric);
    A.compute_cholesky_factorization();

    A.solve(rhs);
    lsq_coeffs = rhs;
  }

  // Perform solution recovery on a C^0-continuous finite element solution field
  // using a modified variant of the Polynomial-Preserving Recovery method of
  // Naga and Zhang. In this version, the patch least-squares problems are
  // implemented using continuous least-squares (L^2 projection). This actually
  // means that the method is not polynomial-preserving, but the overall
  // approach is still based on Naga and Zhang's method.
  //
  // Inputs:
  // - `dof_handler`: The DoFHandler for the finite element solution field to
  //   recover from.
  // - `mapping`: The MappingP1 to map the reference element to physical space.
  //   NOTE: The fact that this is hardcoded to MappingP1 means this function
  //   assumes straight-sided elements. A generalization of this function to
  //   work on curved elements would require this MappingP1 to be changed to
  //   MappingFE initialized with the finite element corresponding to the
  //   finite element solution.
  // - `solution`: The Vector containing the DOFs of the finite element solution
  //   field to recover from.
  // - `dof_handler_enriched`: The DoFHandler for the recovered finite element
  //   solution field.
  //
  // Input/Output:
  // - `solution_enriched`: The Vector where the DOFs of the recovered finite
  //   element solution field will be stored.
  //
  // Reference: A. Naga and Z. Zhang. “A Posteriori Error Estimates Based on the
  // Polynomial Preserving Recovery”. In: SIAM J. Numer. Anal. 42.4 (2004),
  // pp. 1780–1800.
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

    radial::create_vertex_to_cell(dof_handler, vertex_to_cell);

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
      Vector<double> lsq_coeffs(min_points);

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
                                           solution, lsq_coeffs);

      radial::evaluate_patch_polynomial(dof_coords_enriched,
                                        vertex_to_dof_enriched[v],
                                        vertex_to_weight[v],
                                        coord_min, coord_max,
                                        patch_basis_funcs, lsq_coeffs,
                                        solution_enriched);
    }
    //------------------------------------------------------------------------//
  }
} // namespace radial