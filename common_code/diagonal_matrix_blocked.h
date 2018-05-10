
#ifndef diagonal_matrix_blocked_h
#define diagonal_matrix_blocked_h

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>


template <int dim, typename Number>
class DiagonalMatrixBlocked
{
public:
  void vmult(dealii::LinearAlgebra::distributed::BlockVector<Number> &dst,
             const dealii::LinearAlgebra::distributed::BlockVector<Number> &src) const
  {
    AssertThrow(dst.n_blocks()==dim, dealii::ExcNotImplemented());
    for (unsigned int i=0; i<diagonal.local_size(); ++i)
      {
        const Number diag = diagonal.local_element(i);
        for (unsigned int d=0; d<dim; ++d)
          dst.block(d).local_element(i) = diag * src.block(d).local_element(i);
      }
  }

  dealii::LinearAlgebra::distributed::Vector<Number> diagonal;
};

#endif
