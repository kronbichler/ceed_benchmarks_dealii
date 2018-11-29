
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
     * Matrix-vector multiplication including the inner product
     */
    Number vmult_inner_product(LinearAlgebra::distributed::BlockVector<Number> &dst,
                               const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
      accumulated_sum = 0;
      this->data->cell_loop (&MassOperator::local_apply_cell_inner_product,
                             this, dst, src, true);
      return Utilities::MPI::sum(accumulated_sum,
                                 dst.block(0).get_partitioner()->get_mpi_communicator());
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
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> phi(data);
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          phi.reinit (cell);
          phi.read_dof_values(src);
          phi.evaluate (true,false);
          for (unsigned int q=0; q<phi.n_q_points; ++q)
            phi.submit_value (phi.get_value(q), q);
          VectorizedArray<Number> scratch[Utilities::pow(fe_degree+1,dim)*n_components];
          for (unsigned int i=0; i<Utilities::pow(fe_degree+1,dim)*n_components; ++i)
            scratch[i] = phi.begin_dof_values()[i];
          phi.integrate (true,false);
          VectorizedArray<Number> local_sum = VectorizedArray<Number>();
          for (unsigned int i=0; i<Utilities::pow(fe_degree+1,dim)*n_components; ++i)
            local_sum += phi.begin_dof_values()[i] * scratch[i];
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
     * Matrix-vector multiplication including the inner product
     */
    Number vmult_inner_product(LinearAlgebra::distributed::Vector<Number> &dst,
                               const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      accumulated_sum = 0;
      this->data->cell_loop (&MassOperator::local_apply_cell_inner_product,
                             this, dst, src, true);
      return Utilities::MPI::sum(accumulated_sum,
                                 dst.get_partitioner()->get_mpi_communicator());
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
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          phi.reinit (cell);
          phi.read_dof_values(src);
          phi.evaluate (true,false);
          for (unsigned int q=0; q<phi.n_q_points; ++q)
            phi.submit_value (phi.get_value(q), q);
          VectorizedArray<Number> scratch[Utilities::pow(fe_degree+1,dim)];
          for (unsigned int i=0; i<Utilities::pow(fe_degree+1,dim); ++i)
            scratch[i] = phi.begin_dof_values()[i];
          phi.integrate (true,false);
          VectorizedArray<Number> local_sum = VectorizedArray<Number>();
          for (unsigned int i=0; i<Utilities::pow(fe_degree+1,dim); ++i)
            local_sum += phi.begin_dof_values()[i] * scratch[i];
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
