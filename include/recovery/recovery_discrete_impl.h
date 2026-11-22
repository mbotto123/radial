#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/base/point.h>

#include <deal.II/lac/vector.h>

#include <set>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

#include <recovery/recovery_common.h>

// Functions specific to discrete least-squares recovery, including the recovery
// function itself.

namespace radial
{
  using namespace dealii;

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

  // Construct a discrete least-squares problem on a patch of cells, and solve
  // it if the system matrix is well-conditioned enough.
  //
  // This function is specialized to loop over nodes rather than elements. This
  // avoids having to check if a node has already been visited, as is necessary
  // when working in an element-based way. The caveat is that the bases used all
  // need to be nodal.
  //
  // The inputs can be roughly grouped into 4 groups as follows:
  // 1. `dof_coords` and `solution` are "global" vectors, i.e. they contain info
  //    defined over the entire mesh. `dof_coords` has the physical coordinates
  //    of all nodes in the mesh, and `solution` has the finite element solution
  //    value at each node in the mesh. These will be indexed into to extract
  //    the relevant values on the patch we are working on.
  // 2. `patch_dofs` is a set containing the global DOF indices of the DOFs on
  //    the patch. These indices will be used to index into the global vectors
  //    discussed above. The length of `patch_dofs` also provides the number of
  //    rows in the least-squares system matrix. To fully define the dimensions
  //    of the matrix, `min_points` provides the number of columns. `patch_dofs`
  //    must be larger than `min_points` for the system to be solvable.
  // 3. `patch_coord_min`, `patch_coord_max`, and `patch_basis_funcs` provide
  //    information necessary to compute the entries of the least-squares system
  //    matrix. This information includes the coordinates of the bounding box
  //    that contains the patch (used to scale the coordinates of the patch
  //    nodes) as well as the patch basis functions.
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
  void least_squares_patch_discrete(const std::vector<Point<dim>>& dof_coords,
                                    const Vector<double>& solution,
                                    const std::set<types::global_dof_index>& patch_dofs,
                                    const unsigned int& min_points,
                                    const Point<dim>& patch_coord_min,
                                    const Point<dim>& patch_coord_max,
                                    const radial::patch_basis<dim>& patch_basis_funcs,
                                    Vector<double>& lsq_coeffs,
                                    double& rcond)
  {
    // Create RHS and system matrix for discrete least-squares. We use GSL
    // so that condition number estimation can be done once the system
    // matrix is filled.
    gsl_vector *rhs = gsl_vector_alloc(patch_dofs.size());
    gsl_matrix *A = gsl_matrix_alloc(patch_dofs.size(), min_points);

    // Construct least-squares system
    unsigned int eval_count = 0;
    for (const auto& dof : patch_dofs)
    {
      // Sample solution at the patch node
      gsl_vector_set(rhs, eval_count, solution[dof]);

      Point<dim> node_physical_coords = dof_coords[dof];

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

  // Perform solution recovery on a C^0-continuous finite element solution field
  // using the Polynomial-Preserving Recovery method of Naga and Zhang.
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
      Vector<double> lsq_coeffs(min_points);
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
                                             lsq_coeffs, rcond);
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
                                               lsq_coeffs, rcond);
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
                                        patch_basis_funcs, lsq_coeffs,
                                        solution_enriched);
    }
    //------------------------------------------------------------------------//
  }
} // namespace radial
