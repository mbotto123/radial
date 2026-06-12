#include <recovery_common.h>

#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>

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

  // Construct a least-squares problem on a patch of cells, and solve it if the
  // system matrix is well-conditioned enough.
  //
  // The inputs can be roughly grouped into 4 pairs as follows:
  // 1. `patch_cells` and `patch_dofs` provide the details that identify this
  //    patch, i.e. the cells and DOFs contained in the patch. These are
  //    purely input arguments and do not need to be modified.
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
                           const std::vector<std::function<double(Point<dim>)>>& patch_basis_funcs,
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

    Point<dim> coord_min, coord_max;
    radial::find_patch_bounding_box(patch_cells, patch_dofs,
                                    fe_values_nodes, local_dof_indices,
                                    coord_min, coord_max);

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
    double rcond_tol = std::numeric_limits<double>::epsilon();
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
        lsq_coeffs(i) = gsl_vector_get(x, i);

      gsl_vector_free(x);
    }

    gsl_matrix_free(A);
    gsl_matrix_free(T);
    gsl_vector_free(rhs);
  }
} // namespace radial