
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/vector_tools.h>

#include "../bp1/curved_manifold.h"

using namespace dealii;

template <typename Number>
inline DEAL_II_ALWAYS_INLINE
Number do_invert(Tensor<2,2,Number> &t)
{
  const Number det = (t[0][0]*t[1][1]-t[1][0]*t[0][1]);
  const Number inv_det = 1.0/det;
  const Number tmp = inv_det*t[0][0];
  t[0][0] = inv_det*t[1][1];
  t[0][1] = -inv_det*t[0][1];
  t[1][0] = -inv_det*t[1][0];
  t[1][1] = tmp;

  return det;
}


template <typename Number>
inline DEAL_II_ALWAYS_INLINE
Number do_invert (Tensor<2,3,Number> &t)
{
  const Number t4 = t[0][0]*t[1][1],
               t6 = t[0][0]*t[1][2],
               t8 = t[0][1]*t[1][0],
               t00 = t[0][2]*t[1][0],
               t01 = t[0][1]*t[2][0],
               t04 = t[0][2]*t[2][0],
               det = (t4*t[2][2]-t6*t[2][1]-t8*t[2][2]+
                      t00*t[2][1]+t01*t[1][2]-t04*t[1][1]),
               inv_det = 1.0/det;
  const Number tr00 = inv_det*(t[1][1]*t[2][2]-t[1][2]*t[2][1]);
  const Number tr01 = inv_det*(t[0][2]*t[2][1]-t[0][1]*t[2][2]);
  const Number tr02 = inv_det*(t[0][1]*t[1][2]-t[0][2]*t[1][1]);
  const Number tr10 = inv_det*(t[1][2]*t[2][0]-t[1][0]*t[2][2]);
  const Number tr11 = inv_det*(t[0][0]*t[2][2]-t04);
  t[1][2] = inv_det*(t00-t6);
  t[2][0] = inv_det*(t[1][0]*t[2][1]-t[1][1]*t[2][0]);
  t[2][1] = inv_det*(t01-t[0][0]*t[2][1]);
  t[2][2] = inv_det*(t4-t8);
  t[0][0] = tr00;
  t[0][1] = tr01;
  t[0][2] = tr02;
  t[1][0] = tr10;
  t[1][1] = tr11;

  return det;
}


// similar to bp1/bench_mass.cc but now implementing the Laplacian in various
// options.

template <int dim, int fe_degree, int n_q_points_1d = fe_degree+1, int n_components = 1, typename Number = double>
class LaplaceOperator
{
public:
  /**
   * Number typedef.
   */
  typedef Number value_type;

  /**
   * size_type needed for preconditioner classes.
   */
  typedef types::global_dof_index size_type;

  /**
   * Constructor.
   */
  LaplaceOperator () {}

  /**
   * Initialize function.
   */
  void initialize(std::shared_ptr<const MatrixFree<dim,Number> > data_)
  {
    this->data = data_;
    quad_1d = QGauss<1>(n_q_points_1d);
    cell_vertex_coefficients.resize(data->n_macro_cells());
    const std::size_t n_q_points = Utilities::fixed_power<dim>(quad_1d.size());
    merged_coefficients.resize(n_q_points*data->n_macro_cells());
    quadrature_points.resize(n_q_points*dim*data->n_macro_cells());
    FE_Nothing<dim> dummy_fe;
    FEValues<dim> fe_values(dummy_fe, Quadrature<dim>(quad_1d),
                            update_quadrature_points | update_jacobians |
                            update_JxW_values);
    for (unsigned int c=0; c<data->n_macro_cells(); ++c)
      {
        for (unsigned int l=0; l<data->n_components_filled(c); ++l)
          {
            const typename Triangulation<dim>::cell_iterator cell
              = data->get_cell_iterator(c, l);
            if (dim == 2)
              {
                std::array<Tensor<1,dim>, 4> v{{cell->vertex(0), cell->vertex(1),
                                                cell->vertex(2), cell->vertex(3)}};
                for (unsigned int d=0; d<dim; ++d)
                  {
                    cell_vertex_coefficients[c][0][d][l] = v[0][d];
                    cell_vertex_coefficients[c][1][d][l] = v[1][d]-v[0][d];
                    cell_vertex_coefficients[c][2][d][l] = v[2][d]-v[0][d];
                    cell_vertex_coefficients[c][3][d][l] = v[3][d]-v[2][d]-(v[1][d]-v[0][d]);
                  }
              }
            else if (dim==3)
              {
                std::array<Tensor<1,dim>, 8> v{{cell->vertex(0), cell->vertex(1),
                                                cell->vertex(2), cell->vertex(3),
                                                cell->vertex(4), cell->vertex(5),
                                                cell->vertex(6), cell->vertex(7)}};
                for (unsigned int d=0; d<dim; ++d)
                  {
                    cell_vertex_coefficients[c][0][d][l] = v[0][d];
                    cell_vertex_coefficients[c][1][d][l] = v[1][d]-v[0][d];
                    cell_vertex_coefficients[c][2][d][l] = v[2][d]-v[0][d];
                    cell_vertex_coefficients[c][3][d][l] = v[4][d]-v[0][d];
                    cell_vertex_coefficients[c][4][d][l] = v[3][d]-v[2][d]-(v[1][d]-v[0][d]);
                    cell_vertex_coefficients[c][5][d][l] = v[5][d]-v[4][d]-(v[1][d]-v[0][d]);
                    cell_vertex_coefficients[c][6][d][l] = v[6][d]-v[4][d]-(v[2][d]-v[0][d]);
                    cell_vertex_coefficients[c][7][d][l] = (v[7][d]-v[6][d]-(v[5][d]-v[4][d])-
                                                            (v[3][d]-v[2][d]-(v[1][d]-v[0][d])));
                  }
              }
            else
              AssertThrow(false, ExcNotImplemented());

            // compute coefficients for other methods
            fe_values.reinit(cell);
            for (unsigned int q=0; q<n_q_points; ++q)
              {
                Tensor<2,dim> merged_coefficient = fe_values.JxW(q) *
                  (invert(Tensor<2,dim>(fe_values.jacobian(q)))*
                   transpose(invert(Tensor<2,dim>(fe_values.jacobian(q)))));
                for (unsigned int d=0, cc=0; d<dim; ++d)
                  for (unsigned int e=d; e<dim; ++e, ++cc)
                    merged_coefficients[c*n_q_points+q][cc][l] = merged_coefficient[d][e];
                for (unsigned int d=0; d<dim; ++d)
                  quadrature_points[c*dim*n_q_points+d*n_q_points+q][l] = fe_values.quadrature_point(q)[d];
              }

          }
        // insert dummy entries to prevent geometry from degeneration and
        // subsequent division by zero, assuming a Cartesian geometry
        for (unsigned int l=data->n_components_filled(c);
             l<VectorizedArray<Number>::n_array_elements; ++l)
          for (unsigned int d=0; d<dim; ++d)
            cell_vertex_coefficients[c][d+1][d][l] = 1.;
      }
  }

  /**
   * Initialize function.
   */
  void initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> &vec) const
  {
    data->initialize_dof_vector(vec);
  }

  /**
   * Matrix-vector multiplication.
   */
  void vmult(LinearAlgebra::distributed::Vector<Number> &dst,
             const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    dst = 0;
    this->data->cell_loop (&LaplaceOperator::local_apply_basic,
                           this, dst, src);
    for (unsigned int i : data->get_constrained_dofs(0))
      dst.local_element(i) = src.local_element(i);
  }

  /**
   * Matrix-vector multiplication.
   */
  void vmult_linear_geo(LinearAlgebra::distributed::Vector<Number> &dst,
                        const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    dst = 0;
    this->data->cell_loop (&LaplaceOperator::local_apply_linear_geo,
                           this, dst, src);
    for (unsigned int i : data->get_constrained_dofs(0))
      dst.local_element(i) = src.local_element(i);
  }

  /**
   * Matrix-vector multiplication.
   */
  void vmult_merged(LinearAlgebra::distributed::Vector<Number> &dst,
                    const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    dst = 0;
    this->data->cell_loop (&LaplaceOperator::local_apply_merged,
                           this, dst, src);
    for (unsigned int i : data->get_constrained_dofs(0))
      dst.local_element(i) = src.local_element(i);
  }

  /**
   * Matrix-vector multiplication.
   */
  void vmult_construct_q(LinearAlgebra::distributed::Vector<Number> &dst,
                         const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    dst = 0;
    this->data->cell_loop (&LaplaceOperator::local_apply_construct_q,
                           this, dst, src);
    for (unsigned int i : data->get_constrained_dofs(0))
      dst.local_element(i) = src.local_element(i);
  }

  /**
   * Transpose matrix-vector multiplication. Since the Laplace matrix is
   * symmetric, it does exactly the same as vmult().
   */
  void Tvmult(LinearAlgebra::distributed::Vector<Number> &dst,
              const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    vmult(dst, src);
  }

private:
  void
  local_apply_basic (const MatrixFree<dim,value_type>            &data,
                     LinearAlgebra::distributed::Vector<Number>  &dst,
                     const LinearAlgebra::distributed::Vector<Number> &src,
                     const std::pair<unsigned int,unsigned int>  &cell_range) const
  {
    FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> phi(data);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit (cell);
        phi.read_dof_values(src);
        phi.evaluate (false,true);
        for (unsigned int q=0; q<phi.n_q_points; ++q)
          phi.submit_gradient (phi.get_gradient(q), q);
        phi.integrate (false,true);
        phi.distribute_local_to_global (dst);
      }
  }

  void
  local_apply_linear_geo (const MatrixFree<dim,value_type>            &data,
                          LinearAlgebra::distributed::Vector<Number>  &dst,
                          const LinearAlgebra::distributed::Vector<Number> &src,
                          const std::pair<unsigned int,unsigned int>  &cell_range) const
  {
    FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> phi(data);
    constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d,dim);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit (cell);
        phi.read_dof_values(src);
        phi.evaluate (false,true);
        const std::array<Tensor<1,dim,VectorizedArray<Number> >,GeometryInfo<dim>::vertices_per_cell> &v
          = cell_vertex_coefficients[cell];
        VectorizedArray<Number> *phi_grads = phi.begin_gradients();
        if (dim == 2)
          {
            for (unsigned int q=0, qy = 0; qy<n_q_points_1d; ++qy)
              {
                // x-derivative, already complete
                Tensor<1,dim,VectorizedArray<Number>> x_con = v[1] + quad_1d.point(qy)[0]*v[3];
                for (unsigned int qx=0; qx<n_q_points_1d; ++qx, ++q)
                  {
                    const double q_weight = quad_1d.weight(qy) * quad_1d.weight(qx);
                    Tensor<2,dim,VectorizedArray<Number>> jac;
                    jac[1] = v[2] + quad_1d.point(qx)[0]*v[3];
                    jac[0] = x_con;
                    const VectorizedArray<Number> det = do_invert(jac);
                    VectorizedArray<Number> tmp[dim];
                    for (unsigned int d=0; d<dim; ++d)
                      {
                        tmp[d] = jac[d][0] * phi_grads[q];
                        for (unsigned int e=1; e<dim; ++e)
                          tmp[d] += jac[d][e] * phi_grads[q+e*n_q_points];
                        tmp[d] *= det * q_weight;
                      }
                    for (unsigned int d=0; d<dim; ++d)
                      {
                        phi_grads[q+d*phi.n_q_points] = jac[0][d] * tmp[0];
                        for (unsigned int e=1; e<dim; ++e)
                          phi_grads[q+d*phi.n_q_points] += jac[e][d] * tmp[e];
                      }
                  }
              }
          }
        else if (dim==3)
          {
            for (unsigned int q=0, qz = 0; qz<n_q_points_1d; ++qz)
              {
                for (unsigned int qy = 0; qy<n_q_points_1d; ++qy)
                  {
                    // x-derivative, already complete
                    Tensor<1,dim,VectorizedArray<Number>> x_con = v[1] + quad_1d.point(qz)[0]*v[5];
                    x_con += quad_1d.point(qy)[0]*(v[4]+quad_1d.point(qz)[0]*v[7]);
                    // y-derivative, constant part
                    Tensor<1,dim,VectorizedArray<Number>> y_con = v[2] + quad_1d.point(qz)[0]*v[6];
                    // y-derivative, xi-dependent part
                    Tensor<1,dim,VectorizedArray<Number>> y_var = v[4] + quad_1d.point(qz)[0]*v[7];
                    // z-derivative, constant part
                    Tensor<1,dim,VectorizedArray<Number>> z_con = v[3] + quad_1d.point(qy)[0]*v[6];
                    // z-derivative, variable part
                    Tensor<1,dim,VectorizedArray<Number>> z_var = v[5] + quad_1d.point(qy)[0]*v[7];
                    double q_weight_tmp = quad_1d.weight(qz) * quad_1d.weight(qy);
                    for (unsigned int qx=0; qx<n_q_points_1d; ++qx, ++q)
                      {
                        const double q_weight = q_weight_tmp * quad_1d.weight(qx);
                        Tensor<2,dim,VectorizedArray<Number>> jac;
                        jac[1] = y_con + quad_1d.point(qx)[0]*y_var;
                        jac[2] = z_con + quad_1d.point(qx)[0]*z_var;
                        jac[0] = x_con;
                        const VectorizedArray<Number> det = do_invert(jac);

                        VectorizedArray<Number> tmp[dim];
                        for (unsigned int d=0; d<dim; ++d)
                          {
                            tmp[d] = jac[d][0] * phi_grads[q];
                            for (unsigned int e=1; e<dim; ++e)
                              tmp[d] += jac[d][e] * phi_grads[q+e*n_q_points];
                            tmp[d] *= det * q_weight;
                          }
                        for (unsigned int d=0; d<dim; ++d)
                          {
                            phi_grads[q+d*n_q_points] = jac[0][d] * tmp[0];
                            for (unsigned int e=1; e<dim; ++e)
                              phi_grads[q+d*n_q_points] += jac[e][d] * tmp[e];
                          }

                      }
                  }
              }
          }
        phi.integrate (false,true);
        phi.distribute_local_to_global (dst);
      }
  }

  void
  local_apply_merged (const MatrixFree<dim,value_type>            &data,
                      LinearAlgebra::distributed::Vector<Number>  &dst,
                      const LinearAlgebra::distributed::Vector<Number> &src,
                      const std::pair<unsigned int,unsigned int>  &cell_range) const
  {
    FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> phi(data);
    constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit (cell);
        phi.read_dof_values(src);
        phi.evaluate (false,true);
        VectorizedArray<Number> *phi_grads = phi.begin_gradients();
        for (unsigned int q=0; q<n_q_points; ++q)
          {
            if (dim==2)
              {
                VectorizedArray<Number> tmp = phi_grads[q];
                phi_grads[q] = merged_coefficients[cell*n_q_points+q][0] * tmp
                  + merged_coefficients[cell*n_q_points+q][1] * phi_grads[q+n_q_points];
                phi_grads[q+n_q_points] =
                  merged_coefficients[cell*n_q_points+q][1] * tmp
                  + merged_coefficients[cell*n_q_points+q][2] * phi_grads[q+n_q_points];
              }
            else if (dim==3)
              {
                VectorizedArray<Number> tmp0 = phi_grads[q];
                VectorizedArray<Number> tmp1 = phi_grads[q+n_q_points];
                phi_grads[q] =
                  (merged_coefficients[cell*n_q_points+q][0] * tmp0
                   + merged_coefficients[cell*n_q_points+q][1] * tmp1
                   + merged_coefficients[cell*n_q_points+q][2] * phi_grads[q+2*n_q_points]);
                phi_grads[q+n_q_points] =
                  (merged_coefficients[cell*n_q_points+q][1] * tmp0
                   + merged_coefficients[cell*n_q_points+q][3] * tmp1
                   + merged_coefficients[cell*n_q_points+q][4] * phi_grads[q+2*n_q_points]);
                phi_grads[q+2*n_q_points] =
                  (merged_coefficients[cell*n_q_points+q][2] * tmp0
                   + merged_coefficients[cell*n_q_points+q][4] * tmp1
                   + merged_coefficients[cell*n_q_points+q][5] * phi_grads[q+2*n_q_points]);
              }
          }
        phi.integrate (false,true);
        phi.distribute_local_to_global (dst);
      }
  }

  void
  local_apply_construct_q (const MatrixFree<dim,value_type>            &data,
                           LinearAlgebra::distributed::Vector<Number>  &dst,
                           const LinearAlgebra::distributed::Vector<Number> &src,
                           const std::pair<unsigned int,unsigned int>  &cell_range) const
  {
    FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> phi(data);
    constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);
    VectorizedArray<Number> jacobians[dim*dim*n_q_points];
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit (cell);
        phi.read_dof_values(src);
        phi.evaluate (false,true);
        internal::FEEvaluationImplCollocation<dim,n_q_points_1d-1,dim,VectorizedArray<Number>>
          ::evaluate(data.get_shape_info(),
                     quadrature_points.begin()+cell*dim*n_q_points,
                     nullptr, jacobians, nullptr, nullptr,
                     false, true, false);
        VectorizedArray<Number> *phi_grads = phi.begin_gradients();
        for (unsigned int q=0; q<phi.n_q_points; ++q)
          {
            Tensor<2,dim,VectorizedArray<Number>> jac;
            for (unsigned int d=0; d<dim; ++d)
              for (unsigned int e=0; e<dim; ++e)
                jac[d][e] = jacobians[(e*dim+d)*n_q_points+q];
            const VectorizedArray<Number> det = do_invert(jac);
            const Number q_weight = data.get_quadrature().weight(q);

            VectorizedArray<Number> tmp[dim];
            for (unsigned int d=0; d<dim; ++d)
              {
                tmp[d] = jac[d][0] * phi_grads[q];
                for (unsigned int e=1; e<dim; ++e)
                  tmp[d] += jac[d][e] * phi_grads[q+e*n_q_points];
                tmp[d] *= det * q_weight;
              }
            for (unsigned int d=0; d<dim; ++d)
              {
                phi_grads[q+d*n_q_points] = jac[0][d] * tmp[0];
                for (unsigned int e=1; e<dim; ++e)
                  phi_grads[q+d*n_q_points] += jac[e][d] * tmp[e];
              }
          }
        phi.integrate (false,true);
        phi.distribute_local_to_global (dst);
      }
  }

  std::shared_ptr<const MatrixFree<dim,Number> > data;


  Quadrature<1> quad_1d;
  // For local_apply_linear_geo:
  // A list containing the geometry in terms of the bilinear coefficients,
  // i.e. x_0, x_1-x_0, x_2-x_0, x_4-x_0, x_3-x_2-x_1+x_0, ...
  AlignedVector<std::array<Tensor<1,dim,VectorizedArray<Number> >,
                           GeometryInfo<dim>::vertices_per_cell> > cell_vertex_coefficients;

  // For local_apply_merged: dim*(dim+1)/2 coefficients
  AlignedVector<Tensor<1,(dim*(dim+1)/2),VectorizedArray<Number> > > merged_coefficients;

  // for local_apply_construct_q:
  // Data in all quadrature points in collocated space
  AlignedVector<VectorizedArray<Number> > quadrature_points;
};



template <int dim, int fe_degree, int n_q_points>
void test(const unsigned int s,
          const bool short_output)
{
  Timer time;
  const unsigned int n_refine = s/3;
  const unsigned int remainder = s%3;
  Point<dim> p2;
  for (unsigned int d=0; d<remainder; ++d)
    p2[d] = 2;
  for (unsigned int d=remainder; d<dim; ++d)
    p2[d] = 1;

  MyManifold<dim> manifold;
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  std::vector<unsigned int> subdivisions(dim, 1);
  for (unsigned int d=0; d<remainder; ++d)
    subdivisions[d] = 2;
  GridGenerator::subdivided_hyper_rectangle(tria, subdivisions, Point<dim>(), p2);
  GridTools::transform(std::bind(&MyManifold<dim>::push_forward, manifold,
                                 std::placeholders::_1),
                       tria);
  tria.set_all_manifold_ids(1);
  tria.set_manifold(1, manifold);
  tria.refine_global(n_refine);

  FE_Q<dim> fe_q(fe_degree);
  //FESystem<dim> fe(fe_q, dim);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_q);

  ConstraintMatrix constraints;
  IndexSet relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(dof_handler, 0, ZeroFunction<dim>(),
                                           constraints);
  constraints.close();
  std::shared_ptr<MatrixFree<dim,double> > matrix_free(new MatrixFree<dim,double>());
  matrix_free->reinit(dof_handler, constraints, QGauss<1>(n_q_points),
                      typename MatrixFree<dim,double>::AdditionalData());

  LaplaceOperator<dim,fe_degree,n_q_points,1> laplace_operator;
  laplace_operator.initialize(matrix_free);

  LinearAlgebra::distributed::Vector<double> input, output, output_test;
  laplace_operator.initialize_dof_vector(input);
  laplace_operator.initialize_dof_vector(output);
  laplace_operator.initialize_dof_vector(output_test);
  for (unsigned int i=0; i<input.local_size(); ++i)
    if (!constraints.is_constrained(input.get_partitioner()->local_to_global(i)))
      input.local_element(i) = (i)%8;

  Utilities::MPI::MinMaxAvg data =
    Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output == false)
    std::cout << "Setup time:         "
              << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")" << "s"
              << std::endl;

  ReductionControl solver_control(1000, 1e-15, 1e-6);
  SolverCG<LinearAlgebra::distributed::Vector<double> > solver(solver_control);

  time.restart();
  try
    {
      solver.solve(laplace_operator, output, input, PreconditionIdentity());
    }
  catch (SolverControl::NoConvergence &e)
    {
      // prevent the solver to throw an exception in case we should need more
      // than 1000 iterations
    }
  data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output==false)
    {
      std::cout << "Solve time:         "
                << data.min << " (p" << data.min_index << ") " << data.avg
                << " " << data.max << " (p" << data.max_index << ")" << "s"
                << std::endl;
      std::cout << "Time per iteration: "
                << data.min/solver_control.last_step()
                << " (p" << data.min_index << ") "
                << data.avg/solver_control.last_step()
                << " " << data.max/solver_control.last_step()
                << " (p" << data.max_index << ")" << "s"
                << std::endl;
    }
  const double solver_time = data.max;

  time.restart();
  for (unsigned int i=0; i<100; ++i)
    laplace_operator.vmult(input, output);
  const double t1 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max/100;

  time.restart();
  for (unsigned int i=0; i<100; ++i)
    laplace_operator.vmult_merged(output_test, output);
  const double t2 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max/100;
  output_test -= input;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output==false)
    std::cout << "Error merged coefficient tensor:           "
              << output_test.linfty_norm() << std::endl;

  time.restart();
  for (unsigned int i=0; i<100; ++i)
    laplace_operator.vmult_construct_q(output_test, output);
  const double t3 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max/100;
  output_test -= input;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output==false)
    std::cout << "Error collocation evaluation of Jacobian:  "
              << output_test.linfty_norm() << std::endl;

  time.restart();
  for (unsigned int i=0; i<100; ++i)
    laplace_operator.vmult_linear_geo(output_test, output);
  const double t4 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max/100;
  output_test -= input;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output==false)
    std::cout << "Error trilinear interpolation of Jacobian: "
              << output_test.linfty_norm() << std::endl;

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output == true)
    std::cout << std::setw(2) << fe_degree << " | " << std::setw(2) << n_q_points
              << " |" << std::setw(10) << tria.n_global_active_cells()
              << " |" << std::setw(11) << dim*dof_handler.n_dofs()
              << " | " << std::setw(11) << solver_time/solver_control.last_step()
              << " | " << std::setw(11) << dim*dof_handler.n_dofs()/solver_time*solver_control.last_step()
              << " | " << std::setw(4) << solver_control.last_step()
              << " | " << std::setw(11) << t1
              << " | " << std::setw(11) << t2
              << " | " << std::setw(11) << t3
              << " | " << std::setw(11) << t4
              << std::endl;
}


template <int dim, int fe_degree, int n_q_points>
void do_test()
{
  unsigned int s =
    std::max(3U, static_cast<unsigned int>
             (std::log2(1024/fe_degree/fe_degree/fe_degree)));
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << " p |  q | n_element |     n_dofs |     time/it |   dofs/s/it | itCG | time/matvec | timeMVmerge | timeMVcompu | timeMVlinear"
              << std::endl;
  while (Utilities::fixed_power<dim>(fe_degree+1)*(1UL<<s)*dim
         < 6000000ULL*Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    {
      test<dim,fe_degree,n_q_points>(s, true);
      ++s;
    }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << std::endl;
}


int main(int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  do_test<3,1,2>();
  do_test<3,1,3>();
  do_test<3,2,3>();
  do_test<3,2,4>();
  do_test<3,3,5>();
  do_test<3,4,6>();
  do_test<3,5,7>();
  do_test<3,6,8>();
  do_test<3,7,9>();
  do_test<3,8,10>();

  return 0;
}
