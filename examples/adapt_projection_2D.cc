#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_p1.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/base/numbers.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include <set>
#include <functional>

#include <gmsh.h>

#include <solution_recovery.h>

using namespace dealii;

//-------------------------------------------------------------------------//
// L^2 Projection with mesh adaptation based on approximate L^2 error
// control.
//-------------------------------------------------------------------------//

// Function to project from
// DoubleBL in 2D / TripleBL in 3D
//
// Reference for this function:
// Galbraith et al. "Verification of Unstructured Grid Adaptation Components".
// AIAA Journal 58.9 (2020).
template <int dim>
class Solution : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &p,
                                  const unsigned int component = 0) const override;
};

template<int dim>
double Solution<dim>::value(const Point<dim> &p,
                            const unsigned int /*component*/) const
{
  // Advection
  Tensor<1, dim> a;
  for (int d = 0; d < dim; d++)
    a[d] = 1.0;

  // Diffusion
  double nu = 1./30.;

  double mult_term = 1.0;
  for (int d = 0; d < dim; d++)
    mult_term *= (1 - std::exp(-a[d]*(1 - p[d])/nu)) / (1 - std::exp(-a[d]/nu));

  return -mult_term + 1;
}

template<int dim>
Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p,
                                       const unsigned int /*component*/) const
{
  // Advection
  Tensor<1, dim> a;
  for (int d = 0; d < dim; d++)
    a[d] = 1.0;

  // Diffusion
  double nu = 1./30.;

  Tensor<1, dim> num;
  double denom = nu;
  for (int d = 0; d < dim; d++)
  {
    denom *= std::exp(a[d]/nu) - 1;

    num[d] = a[d] * std::exp(a[d]*p[d]/nu);

    for (int dd = 0; dd < dim; dd++)
    {
      // Generally not advisable to have an if statement in a function
      // that gets called a lot. Is there a way to rewrite this without
      // using an if statement?
      if (dd != d)
        num[d] *= std::exp(a[dd]/nu) - std::exp(a[dd]*p[dd]/nu);
    }
  }

  return num / denom;
}

// TODO: move into the radial source code
void make_gmsh_square_model_occ(std::string name)
{
  gmsh::model::add(name);

  gmsh::model::occ::addRectangle(0, 0, 0, 1, 1);

  gmsh::model::occ::synchronize();
}

// TODO: move into the radial source code
void make_gmsh_box_model_occ(std::string name)
{
  gmsh::model::add(name);

  gmsh::model::occ::addBox(0, 0, 0, 1, 1, 1);

  gmsh::model::occ::synchronize();
}

int main()
{
  gmsh::initialize();

  //-------------------------------------------------------------------------//
  // Run parameters

  // Dimension has to be const!
  const int dim = 2;

  int order = 1;

  VectorTools::NormType norm = VectorTools::L2_norm;
  //VectorTools::NormType norm = VectorTools::H1_seminorm;

  // Chamoin and Legoll
  //double e_0 = 1e-3;

  // Geuzaine Gmsh example
  double N = 4000;

  int num_iters = 5;
  //-------------------------------------------------------------------------//

  //-------------------------------------------------------------------------//
  // Adaptation loop

  for (int adapt_iter = 0; adapt_iter < num_iters; adapt_iter++)
  {
    //-------------------------------------------------------------------------//
    // Mesh
    Triangulation<dim> triangulation;

    if (adapt_iter == 0)
    {
      GridGenerator::subdivided_hyper_cube_with_simplices(triangulation, 2);
    }
    else
    {
      GridIn<dim> gridin;
      gridin.attach_triangulation(triangulation);
      std::string mesh_in_name = "square_" + std::to_string(adapt_iter) + ".msh";
      std::ifstream f(mesh_in_name); // read without using Gmsh API
      //std::string f("square.msh"); // read using Gmsh API
      gridin.read_msh(f);
    }
    //-------------------------------------------------------------------------//
  
    //-------------------------------------------------------------------------//
    // Base finite element field
    const FE_SimplexP<dim> fe(order);
    MappingP1<dim> mapping;

    DoFHandler<dim> dof_handler(triangulation); 
    dof_handler.distribute_dofs(fe);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Compute L2 projection
    Vector<double> solution(dof_handler.n_dofs());

    QGaussSimplex<dim> quadrature(order + 2);

    // Empty constraints object (no constraints to enforce)
    AffineConstraints<double> cm;
    cm.close();

    VectorTools::project(dof_handler, cm, quadrature, 
                         Solution<dim>(), solution);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Enriched finite element field
    const int order_enriched = order + 1;
    const FE_SimplexP<dim> fe_enriched(order_enriched);

    DoFHandler<dim> dof_handler_enriched(triangulation); 
    dof_handler_enriched.distribute_dofs(fe_enriched);
    
    Vector<double> solution_enriched(dof_handler_enriched.n_dofs());
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Solution recovery
    radial::recover_solution_ppr(dof_handler, mapping, solution,
                                 dof_handler_enriched, solution_enriched);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Compute L2 projection from enriched solution
    Vector<double> projection_from_enriched(dof_handler.n_dofs());

    Functions::FEFieldFunction<dim> fe_function_enriched(dof_handler_enriched,
                                                         solution_enriched,
                                                         mapping);

    VectorTools::project(dof_handler, cm, quadrature, 
                         fe_function_enriched, projection_from_enriched);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // P1 finite element field for cell size transfer to nodes
    const FE_SimplexP<dim> fe_linear(1);

    FEValues<dim> fe_values_linear(mapping,
                                   fe_linear,
                                   quadrature,
                                   update_values);

    const unsigned int dofs_per_cell_linear = fe_linear.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices_linear(dofs_per_cell_linear);

    DoFHandler<dim> dof_handler_linear(triangulation);
    dof_handler_linear.distribute_dofs(fe_linear);
    //-------------------------------------------------------------------------//
  
    //-------------------------------------------------------------------------//
    // Compute cell-wise approximate L2 error 
    Vector<double> eta(triangulation.n_cells());

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      projection_from_enriched,
                                      fe_function_enriched,
                                      eta,
                                      quadrature, 
                                      norm);

    double eta_global = VectorTools::compute_global_error(triangulation,
                                                          eta,
                                                          norm);
    std::cout << std::endl << "Approximate L2 error = " << eta_global << std::endl;
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // True error (not used for adaptation, just for checking effectivity)
    QWitherdenVincentSimplex<dim> error_quadrature(order_enriched + 2);

    Vector<double> cell_errors(triangulation.n_cells());

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      cell_errors,
                                      error_quadrature,
                                      norm);
    double L2_error = VectorTools::compute_global_error(triangulation,
                                                        cell_errors,
                                                        norm);

    double theta_L2 = eta_global / L2_error;
    std::cout << "theta_L2 = " << theta_L2 << std::endl;
    //-------------------------------------------------------------------------//
    
    //-------------------------------------------------------------------------//
    // Compute global scaling factor for element size formula
    // (WARNING: they add a square root on the squared cell-wise error)
    int rate;
    if (norm == VectorTools::L2_norm)
    {
      rate = order + 1;
    }
    else if (norm == VectorTools::H1_seminorm)
    {
      rate = order;
    }

    double global_factor = 0;
    for (unsigned int i = 0; i < triangulation.n_cells(); i++)
    {
      // Chamoin and Legoll
      //global_factor += std::pow(eta(i), 2.0*dim / (2.0*rate + dim));

      // Geuzaine Gmsh example
      global_factor += std::pow(eta(i), 2.0 / (1.0 + rate));
    }
    // Geuzaine Gmsh example
    global_factor *= (std::pow(rate, (2.0 + rate)/(1.0 + rate)) + std::pow(rate, 1.0/(1.0 + rate)));
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Compute element sizes for current mesh
    Vector<double> h_current(triangulation.n_cells());

    for (const auto &cell: dof_handler.active_cell_iterators())
      h_current(cell->index()) = cell->diameter();
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Loop over elements and compute element size formula for new mesh
    // (WARNING: they add a square root on the squared cell-wise error)
    Vector<double> h_new(triangulation.n_cells());

    for (unsigned int i = 0; i < triangulation.n_cells(); i++)
    {
      // Chamoin and Legoll
      //double local_term = std::pow(eta(i), 2.0 / (2.0*rate + dim));
      //double global_term = std::pow(global_factor, 1.0 / (2.0*rate));

      //double r_K = std::pow(e_0, 1.0/rate) / (local_term * global_term);
      //h_new(i) = r_K * h_current(i);

      // Geuzaine Gmsh example
      double r_K = std::pow(eta(i), 2.0/(2.0 * (1.0 + rate))) *
                   std::pow(rate, 1.0/(dim * (1.0 + rate))) *
                   std::pow((1.0 + rate) * N / global_factor, 1.0/dim);
      h_new(i) = h_current(i) / r_K;
    }
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Transfer element size formula for new mesh to vertices of current mesh

    Vector<double> h_new_nodes(dof_handler_linear.n_dofs());

    DoFTools::distribute_cell_to_dof_vector(dof_handler_linear,
                                            h_new,
                                            h_new_nodes);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Create background mesh for re-meshing via Gmsh API

    int bg_view = gmsh::view::add("background mesh");

    // Number of values needed to define a cell in a Gmsh list-based view, which
    // is 3 coords per vertex plus 1 scalar per vertex. The 3 is hardcoded because
    // Gmsh always expects 3 coords, even if your mesh is 2D.
    int n_vals_per_view_cell = dofs_per_cell_linear * 3 +
                               dofs_per_cell_linear;

    // A Gmsh list-based view is created using one long vector that concatenates
    // all the cell values.
    std::vector<double> view_cells(triangulation.n_cells() * n_vals_per_view_cell);

    for (const auto &cell: dof_handler_linear.active_cell_iterators())
    {
      cell->get_dof_indices(local_dof_indices_linear);
    
      // The order in which the coordinates are listed when using the API is NOT the
      // same as the order when you're writing a .pos file manually. With the API it's
      // (x0,x1,x2,y0,y1,y2,z0,z1,z2). If you write out the .pos file from this view
      // with the API, you can see that the coordinates are not ordered that way in
      // the .pos file.
      for (unsigned int i : fe_values_linear.dof_indices())
      {
        // Offset at which we start inserting values for this cell
        unsigned int cell_start = n_vals_per_view_cell * cell->index();

        // Point default constructor initalizes all coords to zero so this
        // will work for 2D also. Gmsh expects the z-coordinate of all
        // vertices of a 2D mesh to be zero.
        Point<3> vertex; 
        for (int d = 0; d < dim; d++)
          vertex(d) = cell->vertex(i)(d);

        for (int d = 0; d < 3; d++)
          view_cells[cell_start + (d * dofs_per_cell_linear + i)] = vertex(d);
        
        double h_new = h_new_nodes(local_dof_indices_linear[i]);
        view_cells[cell_start + (3 * dofs_per_cell_linear + i)] = h_new;
      }
    }

    // This switch can go in a function templated on dim
    switch (dim)
    {
      case 2:
      {
        gmsh::view::addListData(bg_view, "ST", triangulation.n_cells(), view_cells);
        break;
      }
      case 3:
      {
        gmsh::view::addListData(bg_view, "SS", triangulation.n_cells(), view_cells);
        break;
      }
    }

    // DEBUG
    std::string pos_file_name = "square_" + std::to_string(adapt_iter) + ".pos";
    gmsh::view::write(bg_view, pos_file_name);
    //-------------------------------------------------------------------------//

    //-------------------------------------------------------------------------//
    // Generate new mesh via Gmsh API

    // This switch can go in a function templated on dim
    switch (dim)
    {
      case 2:
      {
        make_gmsh_square_model_occ("square");
        break;
      }
      case 3:
      {
        make_gmsh_box_model_occ("square");
        break;
      }
    }
    
    int bg_field = gmsh::model::mesh::field::add("PostView");

    gmsh::model::mesh::field::setNumber(bg_field, "ViewIndex", bg_view);
    gmsh::model::mesh::field::setAsBackgroundMesh(bg_field);

    // If you don't set these, then Gmsh will only use the size field.
    //
    // If you do set these, then the actual size used by Gmsh is the minimum between
    // the size specified by the size field and the size computed by these default
    // settings. For instance, points have a default size associated with them if you
    // don't specify it. If that default size is smaller than what is specified by
    // your size field, then the size field will be overruled.
    //
    // Maybe not necessarily a bad thing to let Gmsh overrule the size field if the
    // size field gets "too big". Hessian-based adaptation implementations set a max
    // allowable size to account for regions where the Hessian in nearly zero, so in
    // some sense this is similar.
    //
    // So maybe keeping these default settings could be a conservative approach, and
    // the user could be given the option to turn these settings off, with the caveat
    // that then there's no limitation on how large of an element the size field can
    // specify and that could lead to bad meshes. Of course, another option is to
    // implement the max element size limitation in this code and use that instead of
    // Gmsh's defaults.
    //gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    //gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    //gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

    gmsh::model::mesh::generate(dim);
    std::string mesh_out_name = "square_" + std::to_string(adapt_iter + 1) + ".msh";
    gmsh::write(mesh_out_name);

    // Delete everything and start over at the next iteration
    gmsh::clear();
    //-------------------------------------------------------------------------//

    if ((adapt_iter == num_iters - 1) && (dim == 2))
    {
      std::ofstream out("square_final.svg");
      GridOut       grid_out;
      grid_out.write_svg(triangulation, out);
    }

  }
  //-------------------------------------------------------------------------//

  gmsh::finalize();
}
