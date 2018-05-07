
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
  template <int dim, int fe_degree, int n_q_points_1d = fe_degree+1, int n_components = 1,
            typename Number = double>
  class MassOperator
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
     * For this operator, there is just a cell contribution.
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

    std::shared_ptr<const MatrixFree<dim,Number> > data;
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

    std::shared_ptr<const MatrixFree<dim,Number> > data;
  };

}


#endif
