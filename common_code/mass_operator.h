
#ifndef poisson_operator_h
#define poisson_operator_h


#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "solver_cg_optimized.h"

namespace Mass
{
  using namespace dealii;



  // general mass matrix operator for block case
  template <int dim, int fe_degree, int n_q_points_1d = fe_degree+1, int n_components_ = 1,
            typename Number = double>
  class MassOperator
  {
  public:
    /**
     * Number typedef.
     */
    typedef Number value_type;

    /**
     * Make number of components available as variable
     */
    static constexpr unsigned int n_components = n_components_;

    /**
     * size_type needed for preconditioner classes.
     */
    typedef types::global_dof_index size_type;

    /**
     * Constructor.
     */
    MassOperator () {}

    /**
     * Initialize function.
     */
    void initialize(std::shared_ptr<const MatrixFree<dim,Number> > data_)
    {
      this->data = data_;
    }

    /**
     * Initialize function.
     */
    void initialize_dof_vector(LinearAlgebra::distributed::BlockVector<Number> &vec) const
    {
      vec.reinit(dim);
      for (unsigned int d=0; d<dim; ++d)
        data->initialize_dof_vector(vec.block(d));
      vec.collect_sizes();
    }

    /**
     * Matrix-vector multiplication.
     */
    void vmult(LinearAlgebra::distributed::BlockVector<Number> &dst,
               const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
      this->data->cell_loop (&MassOperator::local_apply_cell,
                             this, dst, src, true);
    }

    /**
     * Matrix-vector multiplication including the inner product (locally,
     * without communicating yet)
     */
    Number vmult_inner_product(LinearAlgebra::distributed::BlockVector<Number> &dst,
                               const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
      accumulated_sum = 0;
      this->data->cell_loop (&MassOperator::local_apply_cell_inner_product,
                             this, dst, src, true);
      return accumulated_sum;
    }

    /**
     * Transpose matrix-vector multiplication. Since the mass matrix is
     * symmetric, it does exactly the same as vmult().
     */
    void Tvmult(LinearAlgebra::distributed::BlockVector<Number> &dst,
                const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
      vmult(dst, src);
    }

  private:
    /**
     * Cell contribution for the mass matrix
     */
    void local_apply_cell (const MatrixFree<dim,value_type>            &data,
                           LinearAlgebra::distributed::BlockVector<Number> &dst,
                           const LinearAlgebra::distributed::BlockVector<Number> &src,
                           const std::pair<unsigned int,unsigned int>  &cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> phi(data);
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          phi.reinit (cell);
          phi.read_dof_values(src);
          phi.evaluate (true,false);
          for (unsigned int q=0; q<phi.n_q_points; ++q)
            phi.submit_value (phi.get_value(q), q);
          phi.integrate (true,false);
          phi.distribute_local_to_global (dst);
        }
    }

    /**
     * Like before but including the product between the source and
     * destination
     */
    void
    local_apply_cell_inner_product (const MatrixFree<dim,value_type>            &data,
                                    LinearAlgebra::distributed::BlockVector<Number> &dst,
                                    const LinearAlgebra::distributed::BlockVector<Number> &src,
                                    const std::pair<unsigned int,unsigned int>  &cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> phi_read(data);
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> phi(data);
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          phi.reinit (cell);
          phi_read.reinit(cell);
          phi_read.read_dof_values(src);
          phi.evaluate (true,false);
          for (unsigned int q=0; q<phi.n_q_points; ++q)
            phi.submit_value (phi.get_value(q), q);
          phi.integrate (true,false);
          VectorizedArray<Number> local_sum = VectorizedArray<Number>();
          for (unsigned int i=0; i<Utilities::pow(fe_degree+1,dim)*n_components; ++i)
            local_sum += phi.begin_dof_values()[i] * phi_read.begin_dof_values()[i];
          phi.distribute_local_to_global (dst);
          for (unsigned int v=0; v<data.n_active_entries_per_cell_batch(cell); ++v)
            accumulated_sum += local_sum[v];
        }
    }

    std::shared_ptr<const MatrixFree<dim,Number> > data;
    mutable double accumulated_sum;
  };



  // partial specialization for scalar case with non-block vectors
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  class MassOperator<dim,fe_degree,n_q_points_1d,1,Number>
  {
  public:
    /**
     * Number typedef.
     */
    typedef Number value_type;

    /**
     * Make number of components available as variable
     */
    static constexpr unsigned int n_components = 1;

    /**
     * size_type needed for preconditioner classes.
     */
    typedef types::global_dof_index size_type;

    /**
     * Constructor.
     */
    MassOperator () {}

    /**
     * Initialize function.
     */
    void initialize(std::shared_ptr<const MatrixFree<dim,Number> > data_)
    {
      this->data = data_;
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
    void vmult(LinearAlgebra::distributed::Vector<Number>       &dst,
               const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      this->data->cell_loop (&MassOperator::local_apply_cell,
                             this, dst, src, true);
    }

    /**
     * Matrix-vector multiplication including the inner product (locally,
     * without communicating yet)
     */
    Number vmult_inner_product(LinearAlgebra::distributed::Vector<Number> &dst,
                               const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      accumulated_sum = 0;
      this->data->cell_loop (&MassOperator::local_apply_cell_inner_product,
                             this, dst, src, true);
      return accumulated_sum;
    }

    Tensor<1,7>
    vmult_with_merged_sums(LinearAlgebra::distributed::Vector<Number> &x,
                           LinearAlgebra::distributed::Vector<Number> &g,
                           LinearAlgebra::distributed::Vector<Number> &d,
                           LinearAlgebra::distributed::Vector<Number> &h,
                           const DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>> &prec,
                           const Number alpha,
                           const Number beta) const
    {
      Tensor<1,7,VectorizedArray<Number>> sums;
      this->data->cell_loop(&MassOperator::local_apply_cell_inner_product,
                            this, h, d,
                            [&](const unsigned int start_range,
                                const unsigned int end_range)
                            {
                              do_cg_update4<1,Number,true>(start_range, end_range,
                                                           h.begin(), x.begin(), g.begin(),
                                                           d.begin(),
                                                           prec.get_vector().begin(),
                                                           alpha, beta);
                            },
                            [&](const unsigned int start_range,
                                const unsigned int end_range)
                            {
                              do_cg_update3b<1,Number>(start_range, end_range,
                                                       g.begin(), d.begin(),
                                                       h.begin(), prec.get_vector().begin(),
                                                       sums);
                            });

      dealii::Tensor<1,7> results;
      for (unsigned int i=0; i<7; ++i)
        {
          results[i] = sums[i][0];
          for (unsigned int v=1; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
            results[i] += sums[i][v];
        }
      dealii::Utilities::MPI::sum(dealii::ArrayView<const double>(results.begin_raw(), 7),
                                  d.get_partitioner()->get_mpi_communicator(),
                                  dealii::ArrayView<double>(results.begin_raw(), 7));
      return results;
    }

    /**
     * Transpose matrix-vector multiplication. Since the mass matrix is
     * symmetric, it does exactly the same as vmult().
     */
    void Tvmult(LinearAlgebra::distributed::Vector<Number>       &dst,
                const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      vmult(dst, src);
    }

  private:
    /**
     * For this operator, there is just a cell contribution.
     */
    void local_apply_cell (const MatrixFree<dim,value_type>            &data,
                           LinearAlgebra::distributed::Vector<Number> &dst,
                           const LinearAlgebra::distributed::Vector<Number> &src,
                           const std::pair<unsigned int,unsigned int>  &cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data);
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          phi.reinit (cell);
          phi.read_dof_values(src);
          phi.evaluate (true,false);
          for (unsigned int q=0; q<phi.n_q_points; ++q)
            phi.submit_value (phi.get_value(q), q);
          phi.integrate (true,false);
          phi.distribute_local_to_global (dst);
        }
    }

    /**
     * Like before but including the product between the source and
     * destination
     */
    void
    local_apply_cell_inner_product (const MatrixFree<dim,value_type>            &data,
                                    LinearAlgebra::distributed::Vector<Number>  &dst,
                                    const LinearAlgebra::distributed::Vector<Number> &src,
                                    const std::pair<unsigned int,unsigned int>  &cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data);
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi_read(data);
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          phi.reinit (cell);
          phi_read.reinit(cell);
          phi_read.read_dof_values(src);
          phi.evaluate (phi_read.begin_dof_values(), true, false);
          for (unsigned int q=0; q<phi.n_q_points; ++q)
            phi.submit_value (phi.get_value(q), q);
          phi.integrate (true,false);
          VectorizedArray<Number> local_sum = VectorizedArray<Number>();
          for (unsigned int i=0; i<Utilities::pow(fe_degree+1,dim); ++i)
            local_sum += phi.begin_dof_values()[i] * phi_read.begin_dof_values()[i];
          phi.distribute_local_to_global (dst);
          for (unsigned int v=0; v<data.n_active_entries_per_cell_batch(cell); ++v)
            accumulated_sum += local_sum[v];
        }
    }

    std::shared_ptr<const MatrixFree<dim,Number> > data;
    mutable Number accumulated_sum;
  };

}


#endif
