#ifndef solver_cg_optimized_h
#define solver_cg_optimized_h

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver.h>
#include "diagonal_matrix_blocked.h"



namespace internal
{
  template <typename Number>
  MPI_Comm
  get_communicator(dealii::LinearAlgebra::distributed::Vector<Number> &vec)
  {
    return vec.get_partitioner()->get_mpi_communicator();
  }

  template <typename Number>
  MPI_Comm
  get_communicator(dealii::LinearAlgebra::distributed::BlockVector<Number> &vec)
  {
    return vec.block(0).get_partitioner()->get_mpi_communicator();
  }
}


template <int n_components, typename Number>
dealii::Tensor<1,2>
cg_update1 (dealii::LinearAlgebra::distributed::Vector<Number> &r,
            const dealii::LinearAlgebra::distributed::Vector<Number> &h,
            const dealii::DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<Number>> &preconditioner,
            const Number alpha)
{
  static_assert(n_components == 1, "Only single component support");

  const dealii::LinearAlgebra::distributed::Vector<Number> &prec = preconditioner.get_vector();
  dealii::VectorizedArray<Number> norm_r = dealii::VectorizedArray<Number>(),
    prod_gh = dealii::VectorizedArray<Number>();
  dealii::VectorizedArray<Number> *arr_r = reinterpret_cast<dealii::VectorizedArray<Number>*>(r.begin());
  const dealii::VectorizedArray<Number> *arr_h = reinterpret_cast<const dealii::VectorizedArray<Number>*>(h.begin());
  const dealii::VectorizedArray<Number> *arr_prec = reinterpret_cast<const dealii::VectorizedArray<Number>*>(prec.begin());
  for (unsigned int i=0; i<r.local_size()/dealii::VectorizedArray<Number>::n_array_elements; ++i)
    {
      arr_r[i] += alpha * arr_h[i];
      norm_r += arr_r[i] * arr_r[i];
      prod_gh += arr_r[i] * arr_prec[i] * arr_r[i];
    }
  for (unsigned int i=(r.local_size()/dealii::VectorizedArray<Number>::n_array_elements)*
         dealii::VectorizedArray<Number>::n_array_elements; i<r.local_size(); ++i)
    {
      r.local_element(i) += alpha*h.local_element(i);
      norm_r[0] += r.local_element(i) * r.local_element(i);
      prod_gh[0] += r.local_element(i) * prec.local_element(i) * r.local_element(i);
    }
  dealii::Tensor<1,2> results;
  results[0] = norm_r[0];
  for (unsigned int v=1; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
    results[0] += norm_r[v];
  results[1] = prod_gh[0];
  for (unsigned int v=1; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
    results[1] += prod_gh[v];
  return dealii::Utilities::MPI::sum(results, internal::get_communicator(r));
}



template <int n_components, typename Number>
dealii::Tensor<1,2>
cg_update1 (dealii::LinearAlgebra::distributed::BlockVector<Number> &r,
            const dealii::LinearAlgebra::distributed::BlockVector<Number> &h,
            const DiagonalMatrixBlocked<n_components,Number> &preconditioner,
            const Number alpha)
{
  const dealii::LinearAlgebra::distributed::Vector<Number> &prec = preconditioner.diagonal;
  dealii::VectorizedArray<Number> norm_r = dealii::VectorizedArray<Number>(),
    prod_gh = dealii::VectorizedArray<Number>();
  dealii::VectorizedArray<Number> *arr_r[n_components];
  for (unsigned int d=0; d<n_components; ++d)
    arr_r[d] = reinterpret_cast<dealii::VectorizedArray<Number>*>(r.block(d).begin());
  const dealii::VectorizedArray<Number> *arr_h[n_components];
  for (unsigned int d=0; d<n_components; ++d)
    arr_h[d] = reinterpret_cast<const dealii::VectorizedArray<Number>*>(h.block(d).begin());
  const dealii::VectorizedArray<Number> *arr_prec = reinterpret_cast<const dealii::VectorizedArray<Number>*>(prec.begin());
  for (unsigned int i=0; i<r.block(0).local_size()/dealii::VectorizedArray<Number>::n_array_elements; ++i)
    for (unsigned int d=0; d<n_components; ++d)
      {
        arr_r[d][i] += alpha * arr_h[d][i];
        norm_r += arr_r[d][i] * arr_r[d][i];
        prod_gh += arr_r[d][i] * arr_prec[i] * arr_r[d][i];
      }
  for (unsigned int i=(r.block(0).local_size()/dealii::VectorizedArray<Number>::n_array_elements)*
         dealii::VectorizedArray<Number>::n_array_elements; i<r.block(0).local_size(); ++i)
    for (unsigned int d=0; d<n_components; ++d)
      {
        r.block(d).local_element(i) += alpha*h.block(d).local_element(i);
        norm_r[0] += r.block(d).local_element(i) * r.block(d).local_element(i);
        prod_gh[0] += r.block(d).local_element(i) * prec.local_element(i) * r.block(d).local_element(i);
      }
  dealii::Tensor<1,2> results;
  results[0] = norm_r[0];
  for (unsigned int v=1; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
    results[0] += norm_r[v];
  results[1] = prod_gh[0];
  for (unsigned int v=1; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
    results[1] += prod_gh[v];
  return dealii::Utilities::MPI::sum(results, internal::get_communicator(r));
}



template <int n_components, typename Number>
void cg_update2 (dealii::LinearAlgebra::distributed::Vector<Number> &x,
                 const dealii::LinearAlgebra::distributed::Vector<Number> &r,
                 dealii::LinearAlgebra::distributed::Vector<Number> &p,
                 const dealii::DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<Number>> &preconditioner,
                 const Number alpha,
                 const Number beta)
{
  static_assert(n_components == 1, "Only single component support");
  const dealii::LinearAlgebra::distributed::Vector<Number> &prec = preconditioner.get_vector();

  dealii::VectorizedArray<Number> *arr_p = reinterpret_cast<dealii::VectorizedArray<Number>*>(p.begin());
  dealii::VectorizedArray<Number> *arr_x = reinterpret_cast<dealii::VectorizedArray<Number>*>(x.begin());
  const dealii::VectorizedArray<Number> *arr_r = reinterpret_cast<const dealii::VectorizedArray<Number>*>(r.begin());
  const dealii::VectorizedArray<Number> *arr_prec = reinterpret_cast<const dealii::VectorizedArray<Number>*>(prec.begin());
  for (unsigned int i=0; i<r.local_size()/dealii::VectorizedArray<Number>::n_array_elements; ++i)
    {
      arr_x[i] += alpha * arr_p[i];
      arr_p[i] = beta * arr_p[i] - arr_prec[i] * arr_r[i];
    }
  for (unsigned int i=(r.local_size()/dealii::VectorizedArray<Number>::n_array_elements)*dealii::VectorizedArray<Number>::n_array_elements; i<r.local_size(); ++i)
    {
      x.local_element(i) += alpha * p.local_element(i);
      p.local_element(i) = beta * p.local_element(i) - prec.local_element(i) * r.local_element(i);
    }
}



template <int n_components, typename Number>
void cg_update2 (dealii::LinearAlgebra::distributed::BlockVector<Number> &x,
                 const dealii::LinearAlgebra::distributed::BlockVector<Number> &r,
                 dealii::LinearAlgebra::distributed::BlockVector<Number> &p,
                 const DiagonalMatrixBlocked<n_components,Number> &preconditioner,
                 const Number alpha,
                 const Number beta)
{
  const dealii::LinearAlgebra::distributed::Vector<Number> &prec = preconditioner.diagonal;

  const dealii::VectorizedArray<Number> *arr_r[n_components];
  for (unsigned int d=0; d<n_components; ++d)
    arr_r[d] = reinterpret_cast<const dealii::VectorizedArray<Number>*>(r.block(d).begin());
  dealii::VectorizedArray<Number> *arr_x[n_components];
  for (unsigned int d=0; d<n_components; ++d)
    arr_x[d] = reinterpret_cast<dealii::VectorizedArray<Number>*>(x.block(d).begin());
  dealii::VectorizedArray<Number> *arr_p[n_components];
  for (unsigned int d=0; d<n_components; ++d)
    arr_p[d] = reinterpret_cast<dealii::VectorizedArray<Number>*>(p.block(d).begin());
  const dealii::VectorizedArray<Number> *arr_prec = reinterpret_cast<const dealii::VectorizedArray<Number>*>(prec.begin());
  for (unsigned int i=0; i<r.block(0).local_size()/dealii::VectorizedArray<Number>::n_array_elements; ++i)
    for (unsigned int d=0; d<n_components; ++d)
      {
        arr_x[d][i] += alpha * arr_p[d][i];
        arr_p[d][i] = beta * arr_p[d][i] - arr_prec[i] * arr_r[d][i];
      }
  for (unsigned int i=(r.block(0).local_size()/dealii::VectorizedArray<Number>::n_array_elements)*dealii::VectorizedArray<Number>::n_array_elements; i<r.block(0).local_size(); ++i)
    for (unsigned int d=0; d<n_components; ++d)
      {
        x.block(d).local_element(i) += alpha * p.block(d).local_element(i);
        p.block(d).local_element(i) = beta * p.block(d).local_element(i) - prec.local_element(i) * r.block(d).local_element(i);
      }
}


template <typename VectorType>
class SolverCGOptimized : public dealii::Solver<VectorType>
{
public:
  /**
   * Declare type for container size.
   */
  using size_type = dealii::types::global_dof_index;

  /**
   * Constructor.
   */
  SolverCGOptimized(dealii::SolverControl &           cn)
    : dealii::Solver<VectorType>(cn)
  {}


  /**
   * Virtual destructor.
   */
  virtual ~SolverCGOptimized() override = default;

  /**
   * Solve the linear system $Ax=b$ for x.
   */
  template <typename MatrixType, typename PreconditionerType>
  void
  solve(const MatrixType &        A,
        VectorType &              x,
        const VectorType &        b,
        const PreconditionerType &preconditioner)
  {
    dealii::SolverControl::State conv = dealii::SolverControl::iterate;
    using number = typename VectorType::value_type;

    // Memory allocation
    typename dealii::VectorMemory<VectorType>::Pointer g_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer d_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer h_pointer(this->memory);

    // define some aliases for simpler access
    VectorType &g = *g_pointer;
    VectorType &d = *d_pointer;
    VectorType &h = *h_pointer;

    int    it  = 0;
    double res = -std::numeric_limits<double>::max();

    // resize the vectors, but do not set
    // the values since they'd be overwritten
    // soon anyway.
    g.reinit(x, true);
    d.reinit(x, true);
    h.reinit(x, true);

    number gh, beta;

    // compute residual. if vector is zero, then short-circuit the full
    // computation
    if (!x.all_zero())
      {
        A.vmult(g, x);
        g.add(-1., b);
      }
    else
      g.equ(-1., b);
    res = g.l2_norm();

    conv = this->iteration_status(0, res, x);
    if (conv != dealii::SolverControl::iterate)
      return;

    preconditioner.vmult(h, g);

    d.equ(-1., h);

    gh = g * h;

    while (conv == dealii::SolverControl::iterate)
      {
        it++;
        number alpha = A.vmult_inner_product(h, d);
        alpha = dealii::Utilities::MPI::sum(alpha, internal::get_communicator(d));

        Assert(std::abs(alpha) != 0., dealii::ExcDivideByZero());
        alpha = gh / alpha;

        const dealii::Tensor<1,2> results = cg_update1<MatrixType::n_components,number>(g, h, preconditioner, alpha);

        res = std::sqrt(results[0]);
        beta = gh;
        Assert(std::abs(beta) != 0., dealii::ExcDivideByZero());
        gh = results[1];
        beta = gh / beta;

        conv = this->iteration_status(it, res, x);
        if (conv != dealii::SolverControl::iterate)
          {
            x.add(alpha, d);
            break;
          }

        cg_update2<MatrixType::n_components,number>(x, g, d, preconditioner, alpha, beta);
      }

    // in case of failure: throw exception
    if (conv != dealii::SolverControl::success)
      AssertThrow(false, dealii::SolverControl::NoConvergence(it, res));
    // otherwise exit as normal
  }
};



template <int n_components, typename Number>
dealii::Tensor<1,7>
cg_update3 (const Number sum_hd,
            const dealii::LinearAlgebra::distributed::Vector<Number> &r,
            const dealii::LinearAlgebra::distributed::Vector<Number> &h,
            const dealii::DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<Number>> &preconditioner)
{
  static_assert(n_components == 1, "Only single component support");

  const dealii::LinearAlgebra::distributed::Vector<Number> &prec = preconditioner.get_vector();
  dealii::VectorizedArray<Number> sums[6];
  for (unsigned int i=0; i<6; ++i)
    sums[i] = 0.;
  const dealii::VectorizedArray<Number> *arr_r = reinterpret_cast<const dealii::VectorizedArray<Number>*>(r.begin());
  const dealii::VectorizedArray<Number> *arr_h = reinterpret_cast<const dealii::VectorizedArray<Number>*>(h.begin());
  const dealii::VectorizedArray<Number> *arr_prec = reinterpret_cast<const dealii::VectorizedArray<Number>*>(prec.begin());
  for (unsigned int i=0; i<r.local_size()/dealii::VectorizedArray<Number>::n_array_elements; ++i)
    {
      sums[0] += arr_h[i] * arr_h[i];
      sums[1] += arr_r[i] * arr_h[i];
      sums[2] += arr_r[i] * arr_r[i];
      sums[5] += arr_r[i] * arr_prec[i] * arr_r[i];
      const dealii::VectorizedArray<Number> zi = arr_prec[i] * arr_h[i];
      sums[3] += arr_r[i] * zi;
      sums[4] += arr_h[i] * zi;
    }
  for (unsigned int i=(r.local_size()/dealii::VectorizedArray<Number>::n_array_elements)*
         dealii::VectorizedArray<Number>::n_array_elements; i<r.local_size(); ++i)
    {
      sums[0][0] += h.local_element(i) * h.local_element(i);
      sums[1][0] += r.local_element(i) * h.local_element(i);
      sums[2][0] += r.local_element(i) * r.local_element(i);
      sums[5][0] += r.local_element(i) * prec.local_element(i) * r.local_element(i);
      const Number zi = prec.local_element(i) * h.local_element(i);
      sums[3][0] += r.local_element(i) * zi;
      sums[4][0] += h.local_element(i) * zi;
    }
  dealii::Tensor<1,7> results;
  results[0] = sum_hd;
  for (unsigned int i=1; i<7; ++i)
    {
      results[i] = sums[i-1][0];
      for (unsigned int v=1; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
        results[i] += sums[i-1][v];
    }
  dealii::Utilities::MPI::sum(dealii::ArrayView<const double>(results.begin_raw(), 7),
                              r.get_partitioner()->get_mpi_communicator(),
                              dealii::ArrayView<double>(results.begin_raw(), 7));
  return results;
}



template <int n_components, typename Number>
dealii::Tensor<1,7>
cg_update3b (const dealii::LinearAlgebra::distributed::Vector<Number> &r,
             const dealii::LinearAlgebra::distributed::Vector<Number> &d,
             const dealii::LinearAlgebra::distributed::Vector<Number> &h,
             const dealii::DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<Number>> &preconditioner)
{
  static_assert(n_components == 1, "Only single component support");

  const dealii::LinearAlgebra::distributed::Vector<Number> &prec = preconditioner.get_vector();
  dealii::VectorizedArray<Number> sums[9];
  for (unsigned int i=0; i<9; ++i)
    sums[i] = 0.;
  const dealii::VectorizedArray<Number> *arr_r = reinterpret_cast<const dealii::VectorizedArray<Number>*>(r.begin());
  const dealii::VectorizedArray<Number> *arr_d = reinterpret_cast<const dealii::VectorizedArray<Number>*>(d.begin());
  const dealii::VectorizedArray<Number> *arr_h = reinterpret_cast<const dealii::VectorizedArray<Number>*>(h.begin());
  const dealii::VectorizedArray<Number> *arr_prec = reinterpret_cast<const dealii::VectorizedArray<Number>*>(prec.begin());
  for (unsigned int i=0; i<r.local_size()/dealii::VectorizedArray<Number>::n_array_elements; ++i)
    {
      sums[0] += arr_d[i] * arr_h[i];
      sums[1] += arr_h[i] * arr_h[i];
      sums[2] += arr_r[i] * arr_h[i];
      sums[3] += arr_r[i] * arr_r[i];
      sums[6] += arr_r[i] * arr_prec[i] * arr_r[i];
      const dealii::VectorizedArray<Number> zi = arr_prec[i] * arr_h[i];
      sums[4] += arr_r[i] * zi;
      sums[5] += arr_h[i] * zi;
    }
  for (unsigned int i=(r.local_size()/dealii::VectorizedArray<Number>::n_array_elements)*
         dealii::VectorizedArray<Number>::n_array_elements; i<r.local_size(); ++i)
    {
      sums[0][0] += d.local_element(i) * h.local_element(i);
      sums[1][0] += h.local_element(i) * h.local_element(i);
      sums[2][0] += r.local_element(i) * h.local_element(i);
      sums[3][0] += r.local_element(i) * r.local_element(i);
      sums[6][0] += r.local_element(i) * prec.local_element(i) * r.local_element(i);
      const Number zi = prec.local_element(i) * h.local_element(i);
      sums[4][0] += r.local_element(i) * zi;
      sums[5][0] += h.local_element(i) * zi;
    }
  dealii::Tensor<1,7> results;
  for (unsigned int i=0; i<7; ++i)
    {
      results[i] = sums[i][0];
      for (unsigned int v=1; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
        results[i] += sums[i][v];
    }
  dealii::Utilities::MPI::sum(dealii::ArrayView<const double>(results.begin_raw(), 7),
                              r.get_partitioner()->get_mpi_communicator(),
                              dealii::ArrayView<double>(results.begin_raw(), 7));
  return results;
}



template <int n_components, typename Number>
void cg_update4 (const dealii::LinearAlgebra::distributed::Vector<Number> &h,
                 dealii::LinearAlgebra::distributed::Vector<Number> &x,
                 dealii::LinearAlgebra::distributed::Vector<Number> &r,
                 dealii::LinearAlgebra::distributed::Vector<Number> &p,
                 const dealii::DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<Number>> &preconditioner,
                 const Number alpha,
                 const Number beta)
{
  static_assert(n_components == 1, "Only single component support");
  const dealii::LinearAlgebra::distributed::Vector<Number> &prec = preconditioner.get_vector();

  dealii::VectorizedArray<Number> *arr_p = reinterpret_cast<dealii::VectorizedArray<Number>*>(p.begin());
  dealii::VectorizedArray<Number> *arr_x = reinterpret_cast<dealii::VectorizedArray<Number>*>(x.begin());
  dealii::VectorizedArray<Number> *arr_r = reinterpret_cast<dealii::VectorizedArray<Number>*>(r.begin());
  const dealii::VectorizedArray<Number> *arr_h = reinterpret_cast<const dealii::VectorizedArray<Number>*>(h.begin());
  const dealii::VectorizedArray<Number> *arr_prec = reinterpret_cast<const dealii::VectorizedArray<Number>*>(prec.begin());

  // run loops backwards because the summation loop was run through in forward
  // direction
  for (unsigned int i=(r.local_size()/dealii::VectorizedArray<Number>::n_array_elements)*dealii::VectorizedArray<Number>::n_array_elements; i<r.local_size(); ++i)
    {
      x.local_element(i) += alpha * p.local_element(i);
      r.local_element(i) += alpha * h.local_element(i);
      p.local_element(i) = beta * p.local_element(i) - prec.local_element(i) * r.local_element(i);
    }
  for (int i=r.local_size()/dealii::VectorizedArray<Number>::n_array_elements-1; i>=0; --i)
    {
      arr_x[i] += alpha * arr_p[i];
      arr_r[i] += alpha * arr_h[i];
      arr_p[i] = beta * arr_p[i] - arr_prec[i] * arr_r[i];
    }
}



template <typename VectorType>
class SolverCGOptimizedAllreduce : public dealii::Solver<VectorType>
{
public:
  /**
   * Declare type for container size.
   */
  using size_type = dealii::types::global_dof_index;

  /**
   * Constructor.
   */
  SolverCGOptimizedAllreduce(dealii::SolverControl &           cn)
    : dealii::Solver<VectorType>(cn)
  {}


  /**
   * Virtual destructor.
   */
  virtual ~SolverCGOptimizedAllreduce() override = default;

  /**
   * Solve the linear system $Ax=b$ for x.
   */
  template <typename MatrixType, typename PreconditionerType>
  void
  solve(const MatrixType &        A,
        VectorType &              x,
        const VectorType &        b,
        const PreconditionerType &preconditioner)
  {
    dealii::SolverControl::State conv = dealii::SolverControl::iterate;
    using number = typename VectorType::value_type;

    // Memory allocation
    typename dealii::VectorMemory<VectorType>::Pointer g_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer d_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer h_pointer(this->memory);

    // define some aliases for simpler access
    VectorType &g = *g_pointer;
    VectorType &d = *d_pointer;
    VectorType &h = *h_pointer;

    int    it  = 0;
    double res_norm = -std::numeric_limits<double>::max();

    // resize the vectors, but do not set the values since they'd be
    // overwritten soon anyway.
    g.reinit(x, true);
    d.reinit(x, true);
    h.reinit(x, true);

    // compute residual. if vector is zero, then short-circuit the full
    // computation
    if (!x.all_zero())
      {
        A.vmult(g, x);
        g.add(-1., b);
      }
    else
      g.equ(-1., b);
    res_norm = g.l2_norm();

    conv = this->iteration_status(0, res_norm, x);
    if (conv != dealii::SolverControl::iterate)
      return;

    preconditioner.vmult(h, g);

    d.equ(-1., h);

    while (conv == dealii::SolverControl::iterate)
      {
        it++;

#if 0
        A.vmult(h,d);
        const dealii::Tensor<1,7> results =
          cg_update3b<MatrixType::n_components,number>(g, d, h, preconditioner);
#else
        const number loc_sum_hd = A.vmult_inner_product(h, d);
        const dealii::Tensor<1,7> results =
          cg_update3<MatrixType::n_components,number>(loc_sum_hd, g, h, preconditioner);
#endif

        Assert(std::abs(results[0]) != 0., dealii::ExcDivideByZero());
        const number alpha = results[6] / results[0];

        res_norm = std::sqrt(results[3] + 2*alpha*results[2] + alpha*alpha*results[1]);
        conv = this->iteration_status(it, res_norm, x);
        if (conv != dealii::SolverControl::iterate)
          {
            x.add(alpha, d);
            break;
          }

        // Polak-Ribiere like formula to update
        // r^{k+1}^T M^{-1} * (alpha h) = alpha (r^k + alpha h)^T M^{-1} h
        const number gh = alpha*(results[4] + alpha * results[5]);
        const number beta = gh/results[6];

        cg_update4<MatrixType::n_components,number>(h, x, g, d, preconditioner, alpha, beta);
      }

    // in case of failure: throw exception
    if (conv != dealii::SolverControl::success)
      AssertThrow(false, dealii::SolverControl::NoConvergence(it, res_norm));
    // otherwise exit as normal
  }
};





#endif
