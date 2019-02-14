#ifndef solver_cg_optimized_h
#define solver_cg_optimized_h

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver.h>
#include "diagonal_matrix_blocked.h"

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
  return dealii::Utilities::MPI::sum(results, r.get_partitioner()->get_mpi_communicator());
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
  return dealii::Utilities::MPI::sum(results, r.block(0).get_partitioner()->get_mpi_communicator());
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
dealii::Tensor<1,9>
cg_update3 (const dealii::LinearAlgebra::distributed::Vector<Number> &r,
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
      sums[8] += arr_r[i] * arr_prec[i] * arr_r[i];
      const dealii::VectorizedArray<Number> zi = arr_prec[i] * arr_h[i];
      // Kahan summation part 1
      const dealii::VectorizedArray<Number> tmp1 = arr_r[i] * zi - sums[6];
      const dealii::VectorizedArray<Number> tmp2 = sums[4] + tmp1;
      sums[6] = (tmp2 - sums[4]) - tmp1;
      sums[4] = tmp2;
      // Kahan summation part 2
      const dealii::VectorizedArray<Number> tmp3 = arr_h[i] * zi - sums[7];
      const dealii::VectorizedArray<Number> tmp4 = sums[5] + tmp3;
      sums[7] = (tmp4 - sums[5]) - tmp3;
      sums[5] = tmp4;
    }
  for (unsigned int i=(r.local_size()/dealii::VectorizedArray<Number>::n_array_elements)*
         dealii::VectorizedArray<Number>::n_array_elements; i<r.local_size(); ++i)
    {
      sums[0][0] += d.local_element(i) * h.local_element(i);
      sums[1][0] += h.local_element(i) * h.local_element(i);
      sums[2][0] += r.local_element(i) * h.local_element(i);
      sums[3][0] += r.local_element(i) * r.local_element(i);
      sums[8][0] += r.local_element(i) * prec.local_element(i) * r.local_element(i);
      const Number zi = prec.local_element(i) * h.local_element(i);
      // Kahan summation part 1
      const Number tmp1 = r.local_element(i) * zi - sums[6][0];
      const Number tmp2 = sums[4][0] + tmp1;
      sums[6][0] = (tmp2 - sums[4][0]) - tmp1;
      sums[4][0] = tmp2;
      // Kahan summation part 2
      const Number tmp3 = h.local_element(i) * zi - sums[7][0];
      const Number tmp4 = sums[5][0] + tmp3;
      sums[7][0] = (tmp4 - sums[5][0]) - tmp3;
      sums[5][0] = tmp4;
    }
  dealii::Tensor<1,9> results;
  for (unsigned int i : {0,1,2,3,8})
    {
      results[i] = sums[i][0];
      for (unsigned int v=1; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
        results[i] += sums[i][v];
    }
  for (unsigned int i=4; i<6; ++i)
    {
      results[i] = sums[i][0];
      results[i+2] = sums[i+2][0];
      for (unsigned int v=1; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
        {
          double tmp1 = sums[i][v] - results[i+2];
          double tmp2 = results[i] + tmp1;
          results[i+2] = -sums[i+2][v] + ((tmp2 - results[i]) - tmp1);
          results[i] = tmp2;
        }
    }
  dealii::Utilities::MPI::sum(dealii::ArrayView<const double>(results.begin_raw(), 9),
                              r.get_partitioner()->get_mpi_communicator(),
                              dealii::ArrayView<double>(results.begin_raw(), 9));
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
  for (unsigned int i=0; i<r.local_size()/dealii::VectorizedArray<Number>::n_array_elements; ++i)
    {
      arr_x[i] += alpha * arr_p[i];
      arr_r[i] += alpha * arr_h[i];
      arr_p[i] = beta * arr_p[i] - arr_prec[i] * arr_r[i];
    }
  for (unsigned int i=(r.local_size()/dealii::VectorizedArray<Number>::n_array_elements)*dealii::VectorizedArray<Number>::n_array_elements; i<r.local_size(); ++i)
    {
      x.local_element(i) += alpha * p.local_element(i);
      r.local_element(i) += alpha * h.local_element(i);
      p.local_element(i) = beta * p.local_element(i) - prec.local_element(i) * r.local_element(i);
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

        A.vmult(h, d);

        const dealii::Tensor<1,9> results = cg_update3<MatrixType::n_components,number>(g, d, h, preconditioner);

        Assert(std::abs(results[0]) != 0., dealii::ExcDivideByZero());
        const number alpha = results[8] / results[0];

        res_norm = std::sqrt(results[3] + 2*alpha*results[2] + alpha*alpha*results[1]);
        conv = this->iteration_status(it, res_norm, x);
        if (conv != dealii::SolverControl::iterate)
          {
            x.add(alpha, d);
            break;
          }

        // Polak-Ribiere like formula to update
        // r^{k+1}^T M^{-1} * (alpha h) = alpha (r^k + alpha h)^T M^{-1} h
        //
        // rather than using results[4] + alpha*results[5] directly, we
        // use the ingredients present in the Kahan summation that we got
        // from above
        const number y = alpha*results[5] - results[6];
        const number rk_plus_alpha_h = (results[4] + y) + alpha*results[7];
        const number gh = alpha*rk_plus_alpha_h;
        const number beta = gh/results[8];

        cg_update4<MatrixType::n_components,number>(h, x, g, d, preconditioner, alpha, beta);
      }

    // in case of failure: throw exception
    if (conv != dealii::SolverControl::success)
      AssertThrow(false, dealii::SolverControl::NoConvergence(it, res_norm));
    // otherwise exit as normal
  }
};





#endif
