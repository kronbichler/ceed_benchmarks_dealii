
#ifndef poisson_operator_h
#define poisson_operator_h

#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include "solver_cg_optimized.h"
#include "vector_access_reduced.h"

namespace Poisson
{
  using namespace dealii;

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE Number do_invert(Tensor<2, 2, Number> &t)
  {
    const Number det     = (t[0][0] * t[1][1] - t[1][0] * t[0][1]);
    const Number inv_det = 1.0 / det;
    const Number tmp     = inv_det * t[0][0];
    t[0][0]              = inv_det * t[1][1];
    t[0][1]              = -inv_det * t[0][1];
    t[1][0]              = -inv_det * t[1][0];
    t[1][1]              = tmp;

    return det;
  }


  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE Number do_invert(Tensor<2, 3, Number> &t)
  {
    const Number t4 = t[0][0] * t[1][1], t6 = t[0][0] * t[1][2], t8 = t[0][1] * t[1][0],
                 t00 = t[0][2] * t[1][0], t01 = t[0][1] * t[2][0], t04 = t[0][2] * t[2][0],
                 det = (t4 * t[2][2] - t6 * t[2][1] - t8 * t[2][2] + t00 * t[2][1] + t01 * t[1][2] -
                        t04 * t[1][1]),
                 inv_det = 1.0 / det;
    const Number tr00    = inv_det * (t[1][1] * t[2][2] - t[1][2] * t[2][1]);
    const Number tr01    = inv_det * (t[0][2] * t[2][1] - t[0][1] * t[2][2]);
    const Number tr02    = inv_det * (t[0][1] * t[1][2] - t[0][2] * t[1][1]);
    const Number tr10    = inv_det * (t[1][2] * t[2][0] - t[1][0] * t[2][2]);
    const Number tr11    = inv_det * (t[0][0] * t[2][2] - t04);
    t[1][2]              = inv_det * (t00 - t6);
    t[2][0]              = inv_det * (t[1][0] * t[2][1] - t[1][1] * t[2][0]);
    t[2][1]              = inv_det * (t01 - t[0][0] * t[2][1]);
    t[2][2]              = inv_det * (t4 - t8);
    t[0][0]              = tr00;
    t[0][1]              = tr01;
    t[0][2]              = tr02;
    t[1][0]              = tr10;
    t[1][1]              = tr11;

    return det;
  }



  namespace internal
  {
    template <typename MFType, typename Number>
    void
    do_initialize_vector(const std::shared_ptr<const MFType> &       data,
                         LinearAlgebra::distributed::Vector<Number> &vec)
    {
      data->initialize_dof_vector(vec);
    }

    template <typename MFType, typename Number>
    void
    do_initialize_vector(const std::shared_ptr<const MFType> &            data,
                         LinearAlgebra::distributed::BlockVector<Number> &vec)
    {
      for (unsigned int bl = 0; bl < vec.n_blocks(); ++bl)
        data->initialize_dof_vector(vec.block(bl));
      vec.collect_sizes();
    }

    template <typename Number>
    Number
    set_constrained_entries(const std::vector<unsigned int> &                 constrained_entries,
                            const LinearAlgebra::distributed::Vector<Number> &src,
                            LinearAlgebra::distributed::Vector<Number> &      dst)
    {
      Number sum = 0;
      for (unsigned int i = 0; i < constrained_entries.size(); ++i)
        {
          dst.local_element(constrained_entries[i]) = src.local_element(constrained_entries[i]);
          sum +=
            src.local_element(constrained_entries[i]) * src.local_element(constrained_entries[i]);
        }
      return sum;
    }

    template <typename Number>
    Number
    set_constrained_entries(const std::vector<unsigned int> &constrained_entries,
                            const LinearAlgebra::distributed::BlockVector<Number> &src,
                            LinearAlgebra::distributed::BlockVector<Number> &      dst)
    {
      Number sum = 0;
      for (unsigned int bl = 0; bl < dst.n_blocks(); ++bl)
        for (unsigned int i = 0; i < constrained_entries.size(); ++i)
          {
            dst.block(bl).local_element(constrained_entries[i]) =
              src.block(bl).local_element(constrained_entries[i]);
            sum += src.block(bl).local_element(constrained_entries[i]) *
                   src.block(bl).local_element(constrained_entries[i]);
          }
      return sum;
    }
  } // namespace internal



  // similar to bp1/bench_mass.cc but now implementing the Laplacian in various
  // options.

  template <int dim,
            int fe_degree,
            int n_q_points_1d            = fe_degree + 1,
            int n_components_            = 1,
            typename Number              = double,
            typename VectorType          = LinearAlgebra::distributed::Vector<Number>,
            typename VectorizedArrayType = VectorizedArray<Number>>
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
     * Make number of components available as variable
     */
    static constexpr unsigned int n_components = n_components_;

    /**
     * Constructor.
     */
    LaplaceOperator()
    {}

    /**
     * Initialize function.
     */
    void
    initialize(std::shared_ptr<const MatrixFree<dim, Number, VectorizedArrayType>> data_,
               const AffineConstraints<double> &                                   constraints)
    {
      this->data = data_;
      quad_1d    = QGauss<1>(n_q_points_1d);
      cell_vertex_coefficients.resize(data->n_macro_cells());
      const std::size_t n_q_points = Utilities::fixed_power<dim>(quad_1d.size());
      merged_coefficients.resize(n_q_points * data->n_macro_cells());
      quadrature_points.resize(n_q_points * dim * data->n_macro_cells());

      if (fe_degree > 2)
        {
          compressed_dof_indices.resize(Utilities::pow(3, dim) *
                                          VectorizedArrayType::size() *
                                          data->n_macro_cells(),
                                        numbers::invalid_unsigned_int);
          all_indices_uniform.resize(Utilities::pow(3, dim) * data->n_macro_cells(), 1);
        }

      FE_Nothing<dim>                      dummy_fe;
      FEValues<dim>                        fe_values(dummy_fe,
                              Quadrature<dim>(quad_1d),
                              update_quadrature_points | update_jacobians | update_JxW_values);
      std::vector<types::global_dof_index> dof_indices(
        data->get_dof_handler().get_fe().dofs_per_cell);
      for (unsigned int c = 0; c < data->n_macro_cells(); ++c)
        {
          for (unsigned int l = 0; l < data->n_components_filled(c); ++l)
            {
              const typename DoFHandler<dim>::cell_iterator cell = data->get_cell_iterator(c, l);
              if (dim == 2)
                {
                  std::array<Tensor<1, dim>, 4> v{
                    {cell->vertex(0), cell->vertex(1), cell->vertex(2), cell->vertex(3)}};
                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      cell_vertex_coefficients[c][0][d][l] = v[0][d];
                      cell_vertex_coefficients[c][1][d][l] = v[1][d] - v[0][d];
                      cell_vertex_coefficients[c][2][d][l] = v[2][d] - v[0][d];
                      cell_vertex_coefficients[c][3][d][l] =
                        v[3][d] - v[2][d] - (v[1][d] - v[0][d]);
                    }
                }
              else if (dim == 3)
                {
                  std::array<Tensor<1, dim>, 8> v{{cell->vertex(0),
                                                   cell->vertex(1),
                                                   cell->vertex(2),
                                                   cell->vertex(3),
                                                   cell->vertex(4),
                                                   cell->vertex(5),
                                                   cell->vertex(6),
                                                   cell->vertex(7)}};
                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      cell_vertex_coefficients[c][0][d][l] = v[0][d];
                      cell_vertex_coefficients[c][1][d][l] = v[1][d] - v[0][d];
                      cell_vertex_coefficients[c][2][d][l] = v[2][d] - v[0][d];
                      cell_vertex_coefficients[c][3][d][l] = v[4][d] - v[0][d];
                      cell_vertex_coefficients[c][4][d][l] =
                        v[3][d] - v[2][d] - (v[1][d] - v[0][d]);
                      cell_vertex_coefficients[c][5][d][l] =
                        v[5][d] - v[4][d] - (v[1][d] - v[0][d]);
                      cell_vertex_coefficients[c][6][d][l] =
                        v[6][d] - v[4][d] - (v[2][d] - v[0][d]);
                      cell_vertex_coefficients[c][7][d][l] =
                        (v[7][d] - v[6][d] - (v[5][d] - v[4][d]) -
                         (v[3][d] - v[2][d] - (v[1][d] - v[0][d])));
                    }
                }
              else
                AssertThrow(false, ExcNotImplemented());

              // compute coefficients for other methods
              fe_values.reinit(typename Triangulation<dim>::cell_iterator(cell));
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  Tensor<2, dim> merged_coefficient =
                    fe_values.JxW(q) * (invert(Tensor<2, dim>(fe_values.jacobian(q))) *
                                        transpose(invert(Tensor<2, dim>(fe_values.jacobian(q)))));
                  for (unsigned int d = 0, cc = 0; d < dim; ++d)
                    for (unsigned int e = d; e < dim; ++e, ++cc)
                      merged_coefficients[c * n_q_points + q][cc][l] = merged_coefficient[d][e];
                  for (unsigned int d = 0; d < dim; ++d)
                    quadrature_points[c * dim * n_q_points + d * n_q_points + q][l] =
                      fe_values.quadrature_point(q)[d];
                }

              if (fe_degree > 2)
                {
                  cell->get_dof_indices(dof_indices);
                  constexpr unsigned int n_lanes = VectorizedArrayType::size();
                  const unsigned int     offset  = Utilities::pow(3, dim) * (n_lanes * c) + l;
                  const Utilities::MPI::Partitioner &part =
                    *data->get_dof_info().vector_partitioner;
                  unsigned int cc = 0, cf = 0;
                  for (; cf < GeometryInfo<dim>::vertices_per_cell; ++cf, cc += n_lanes)
                    if (!constraints.is_constrained(dof_indices[cf]))
                      compressed_dof_indices[offset + cc] = part.global_to_local(dof_indices[cf]);

                  for (unsigned int line = 0; line < GeometryInfo<dim>::lines_per_cell; ++line)
                    {
                      if (!constraints.is_constrained(dof_indices[cf]))
                        {
                          for (unsigned int i = 0; i < fe_degree - 1; ++i)
                            AssertThrow(dof_indices[cf + i] == dof_indices[cf] + i,
                                        ExcMessage("Expected contiguous numbering"));
                          compressed_dof_indices[offset + cc] =
                            part.global_to_local(dof_indices[cf]);
                        }
                      cc += n_lanes;
                      cf += fe_degree - 1;
                    }
                  for (unsigned int quad = 0; quad < GeometryInfo<dim>::quads_per_cell; ++quad)
                    {
                      if (!constraints.is_constrained(dof_indices[cf]))
                        {
                          for (unsigned int i = 0; i < (fe_degree - 1) * (fe_degree - 1); ++i)
                            AssertThrow(dof_indices[cf + i] == dof_indices[cf] + i,
                                        ExcMessage("Expected contiguous numbering"));
                          compressed_dof_indices[offset + cc] =
                            part.global_to_local(dof_indices[cf]);
                        }
                      cc += n_lanes;
                      cf += (fe_degree - 1) * (fe_degree - 1);
                    }
                  for (unsigned int hex = 0; hex < GeometryInfo<dim>::hexes_per_cell; ++hex)
                    {
                      if (!constraints.is_constrained(dof_indices[cf]))
                        {
                          for (unsigned int i = 0;
                               i < (fe_degree - 1) * (fe_degree - 1) * (fe_degree - 1);
                               ++i)
                            AssertThrow(dof_indices[cf + i] == dof_indices[cf] + i,
                                        ExcMessage("Expected contiguous numbering"));
                          compressed_dof_indices[offset + cc] =
                            part.global_to_local(dof_indices[cf]);
                        }
                      cc += n_lanes;
                      cf += (fe_degree - 1) * (fe_degree - 1) * (fe_degree - 1);
                    }
                  AssertThrow(cc == n_lanes * Utilities::pow(3, dim),
                              ExcMessage("Expected 3^dim dofs, got " + std::to_string(cc)));
                  AssertThrow(cf == dof_indices.size(),
                              ExcMessage("Expected (fe_degree+1)^dim dofs, got " +
                                         std::to_string(cf)));
                }
            }
          // insert dummy entries to prevent geometry from degeneration and
          // subsequent division by zero, assuming a Cartesian geometry
          for (unsigned int l = data->n_components_filled(c);
               l < VectorizedArrayType::size();
               ++l)
            for (unsigned int d = 0; d < dim; ++d)
              cell_vertex_coefficients[c][d + 1][d][l] = 1.;

          if (fe_degree > 2)
            {
              for (unsigned int i = 0; i < Utilities::pow(3, dim); ++i)
                for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
                  if (compressed_dof_indices[Utilities::pow(3, dim) *
                                               (VectorizedArrayType::size() * c) +
                                             i * VectorizedArrayType::size() + v] ==
                      numbers::invalid_unsigned_int)
                    all_indices_uniform[Utilities::pow(3, dim) * c + i] = 0;
            }
        }
    }

    /**
     * Initialize function.
     */
    void
    initialize_dof_vector(VectorType &vec) const
    {
      internal::do_initialize_vector(data, vec);
    }

    /**
     * Matrix-vector multiplication.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      this->data->cell_loop(
        &LaplaceOperator::template local_apply_linear_geo<false>, this, dst, src, true);
      internal::set_constrained_entries(data->get_constrained_dofs(0), src, dst);
    }

    /**
     * Matrix-vector multiplication.
     */
    void
    vmult_add(VectorType &dst, const VectorType &src) const
    {
      this->data->cell_loop(
        &LaplaceOperator::template local_apply_linear_geo<false>, this, dst, src, false);
      internal::set_constrained_entries(data->get_constrained_dofs(0), src, dst);
    }

    /**
     * Matrix-vector multiplication including the inner product (locally,
     * without communicating yet)
     */
    Number
    vmult_inner_product(VectorType &dst, const VectorType &src) const
    {
      accumulated_sum = 0;
      this->data->cell_loop(
        &LaplaceOperator::template local_apply_linear_geo<true>, this, dst, src, true);
      accumulated_sum += internal::set_constrained_entries(data->get_constrained_dofs(0), src, dst);
      return accumulated_sum;
    }

    Tensor<1, 7>
    vmult_with_merged_sums(VectorType &                                                      x,
                           VectorType &                                                      g,
                           VectorType &                                                      d,
                           VectorType &                                                      h,
                           const DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>> &prec,
                           const Number                                                      alpha,
                           const Number                                                      beta,
                           const Number alpha_old,
                           const Number beta_old) const
    {
      Tensor<1, 7, VectorizedArray<Number>> sums;
      this->data->cell_loop(&LaplaceOperator::template local_apply_linear_geo<false>,
                            this,
                            h,
                            d,
                            [&](const unsigned int start_range, const unsigned int end_range) {
                              for (unsigned int bl = 0; bl < ::internal::get_n_blocks(x); ++bl)
                                do_cg_update4b<1, Number, true>(
                                  start_range,
                                  end_range,
                                  ::internal::get_block(h, bl).begin(),
                                  ::internal::get_block(x, bl).begin(),
                                  ::internal::get_block(g, bl).begin(),
                                  ::internal::get_block(d, bl).begin(),
                                  prec.get_vector().begin(),
                                  alpha,
                                  beta,
                                  alpha_old,
                                  beta_old);
                            },
                            [&](const unsigned int start_range, const unsigned int end_range) {
                              for (unsigned int bl = 0; bl < ::internal::get_n_blocks(x); ++bl)
                                do_cg_update3b<1, Number>(start_range,
                                                          end_range,
                                                          ::internal::get_block(g, bl).begin(),
                                                          ::internal::get_block(d, bl).begin(),
                                                          ::internal::get_block(h, bl).begin(),
                                                          prec.get_vector().begin(),
                                                          sums);
                            });

      dealii::Tensor<1, 7> results;
      for (unsigned int i = 0; i < 7; ++i)
        {
          results[i] = sums[i][0];
          for (unsigned int v = 1; v < dealii::VectorizedArray<Number>::size(); ++v)
            results[i] += sums[i][v];
        }
      dealii::Utilities::MPI::sum(
        dealii::ArrayView<const double>(results.begin_raw(), 7),
        ::internal::get_block(g, 0).get_partitioner()->get_mpi_communicator(),
        dealii::ArrayView<double>(results.begin_raw(), 7));
      return results;
    }

    Tensor<1, 7>
    vmult_with_merged_sums(VectorType &                              x,
                           VectorType &                              g,
                           VectorType &                              d,
                           VectorType &                              h,
                           const DiagonalMatrixBlocked<dim, Number> &prec,
                           const Number                              alpha,
                           const Number                              beta,
                           const Number                              alpha_old,
                           const Number                              beta_old) const
    {
      Tensor<1, 7, VectorizedArray<Number>> sums;
      this->data->cell_loop(&LaplaceOperator::template local_apply_linear_geo<false>,
                            this,
                            h,
                            d,
                            [&](const unsigned int start_range, const unsigned int end_range) {
                              for (unsigned int bl = 0; bl < ::internal::get_n_blocks(x); ++bl)
                                do_cg_update4b<1, Number, true>(
                                  start_range,
                                  end_range,
                                  ::internal::get_block(h, bl).begin(),
                                  ::internal::get_block(x, bl).begin(),
                                  ::internal::get_block(g, bl).begin(),
                                  ::internal::get_block(d, bl).begin(),
                                  prec.diagonal.begin(),
                                  alpha,
                                  beta,
                                  alpha_old,
                                  beta_old);
                            },
                            [&](const unsigned int start_range, const unsigned int end_range) {
                              for (unsigned int bl = 0; bl < ::internal::get_n_blocks(x); ++bl)
                                do_cg_update3b<1, Number>(start_range,
                                                          end_range,
                                                          ::internal::get_block(g, bl).begin(),
                                                          ::internal::get_block(d, bl).begin(),
                                                          ::internal::get_block(h, bl).begin(),
                                                          prec.diagonal.begin(),
                                                          sums);
                            });

      dealii::Tensor<1, 7> results;
      for (unsigned int i = 0; i < 7; ++i)
        {
          results[i] = sums[i][0];
          for (unsigned int v = 1; v < dealii::VectorizedArray<Number>::size(); ++v)
            results[i] += sums[i][v];
        }
      dealii::Utilities::MPI::sum(
        dealii::ArrayView<const double>(results.begin_raw(), 7),
        ::internal::get_block(g, 0).get_partitioner()->get_mpi_communicator(),
        dealii::ArrayView<double>(results.begin_raw(), 7));
      return results;
    }

    /*
    std::array<Number,7>
    vmult_with_all_cg_updates(const Number alpha,
                              const Number beta,
                              const DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>>
    &prec, LinearAlgebra::distributed::Vector<Number> &g, LinearAlgebra::distributed::Vector<Number>
    &h, LinearAlgebra::distributed::Vector<Number> &d, LinearAlgebra::distributed::Vector<Number>
    &x) const
    {
      Tensor<1,7,VectorizedArray<Number>> sums;
      this->data->cell_loop(&LaplaceOperator::template local_apply_linear_geo<false>,
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

      std::array<Number,7> results;
      for (unsigned int i=0; i<7; ++i)
        {
          results[i] = sums[i][0];
          for (unsigned int v=1; v<dealii::VectorizedArray<Number>::size(); ++v)
            results[i] += sums[i][v];
        }
      dealii::Utilities::MPI::sum(dealii::ArrayView<const double>(results.data(), 7),
                                  d.get_partitioner()->get_mpi_communicator(),
                                  dealii::ArrayView<double>(results.data(), 7));
      return results;
    }
    */

    /**
     * Matrix-vector multiplication.
     */
    void
    vmult_merged(VectorType &dst, const VectorType &src) const
    {
      this->data->cell_loop(&LaplaceOperator::local_apply_merged, this, dst, src, true);
      internal::set_constrained_entries(data->get_constrained_dofs(0), src, dst);
    }

    /**
     * Matrix-vector multiplication.
     */
    void
    vmult_basic(VectorType &dst, const VectorType &src) const
    {
      this->data->cell_loop(&LaplaceOperator::local_apply_basic, this, dst, src, true);
      internal::set_constrained_entries(data->get_constrained_dofs(0), src, dst);
    }

    /**
     * Matrix-vector multiplication.
     */
    void
    vmult_construct_q(VectorType &dst, const VectorType &src) const
    {
      this->data->cell_loop(&LaplaceOperator::local_apply_construct_q, this, dst, src, true);
      internal::set_constrained_entries(data->get_constrained_dofs(0), src, dst);
    }

    /**
     * Transpose matrix-vector multiplication. Since the Laplace matrix is
     * symmetric, it does exactly the same as vmult().
     */
    void
    Tvmult(VectorType &dst, const VectorType &src) const
    {
      vmult(dst, src);
    }

    /**
     * Compute the diagonal (scalar variant) of the matrix
     */
    LinearAlgebra::distributed::Vector<Number>
    compute_inverse_diagonal() const
    {
      LinearAlgebra::distributed::Vector<Number> diag;
      data->initialize_dof_vector(diag);
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number, VectorizedArrayType> phi(*data);

      AlignedVector<VectorizedArrayType> diagonal(phi.dofs_per_cell);
      for (unsigned int cell = 0; cell < data->n_macro_cells(); ++cell)
        {
          phi.reinit(cell);
          for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                phi.submit_dof_value(VectorizedArrayType(), j);
              phi.submit_dof_value(make_vectorized_array<VectorizedArrayType>(1.), i);

              phi.evaluate(false, true);
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                phi.submit_gradient(phi.get_gradient(q), q);
              phi.integrate(false, true);
              diagonal[i] = phi.get_dof_value(i);
            }
          for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            phi.submit_dof_value(diagonal[i], i);
          phi.distribute_local_to_global(diag);
        }
      diag.compress(VectorOperation::add);
      for (unsigned int i = 0; i < diag.local_size(); ++i)
        if (diag.local_element(i) == 0.)
          diag.local_element(i) = 1.;
        else
          diag.local_element(i) = 1. / diag.local_element(i);
      return diag;
    }

  private:
    void
    local_apply_basic(const MatrixFree<dim, value_type, VectorizedArrayType> &data,
                      VectorType &                                            dst,
                      const VectorType &                                      src,
                      const std::pair<unsigned int, unsigned int> &           cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(src);
          phi.evaluate(false, true);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_gradient(phi.get_gradient(q), q);
          phi.integrate(false, true);
          phi.distribute_local_to_global(dst);
        }
    }

    template <bool do_sum>
    void
    local_apply_linear_geo(const MatrixFree<dim, value_type, VectorizedArrayType> &data,
                           VectorType &                                            dst,
                           const VectorType &                                      src,
                           const std::pair<unsigned int, unsigned int> &           cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        data);
      constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          if (fe_degree > 2)
            for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
              read_dof_values_compressed<dim, fe_degree, value_type>(
                ::internal::get_block(src, bl),
                compressed_dof_indices,
                all_indices_uniform,
                cell,
                phi.begin_dof_values() + bl * Utilities::pow(fe_degree + 1, dim));
          else
            phi.read_dof_values(src);
          phi.evaluate(false, true);
          const std::array<Tensor<1, dim, VectorizedArrayType>,
                           GeometryInfo<dim>::vertices_per_cell> &v =
            cell_vertex_coefficients[cell];
          VectorizedArrayType *phi_grads = phi.begin_gradients();
          if (dim == 2)
            {
              for (unsigned int q = 0, qy = 0; qy < n_q_points_1d; ++qy)
                {
                  // x-derivative, already complete
                  Tensor<1, dim, VectorizedArrayType> x_con = v[1] + quad_1d.point(qy)[0] * v[3];
                  for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                    {
                      const double q_weight = quad_1d.weight(qy) * quad_1d.weight(qx);
                      Tensor<2, dim, VectorizedArrayType> jac;
                      jac[1] = v[2] + quad_1d.point(qx)[0] * v[3];
                      for (unsigned int d = 0; d < 2; ++d)
                        jac[0][d] = x_con[d];
                      const VectorizedArrayType det = do_invert(jac);

                      for (unsigned int c = 0; c < n_components; ++c)
                        {
                          const unsigned int  offset = c * dim * n_q_points;
                          VectorizedArrayType tmp[dim];
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              tmp[d] = jac[d][0] * phi_grads[q + offset];
                              for (unsigned int e = 1; e < dim; ++e)
                                tmp[d] += jac[d][e] * phi_grads[q + e * n_q_points + offset];
                              tmp[d] *= det * q_weight;
                            }
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              phi_grads[q + d * n_q_points + offset] = jac[0][d] * tmp[0];
                              for (unsigned int e = 1; e < dim; ++e)
                                phi_grads[q + d * n_q_points + offset] += jac[e][d] * tmp[e];
                            }
                        }
                    }
                }
            }
          else if (dim == 3)
            {
              for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
                {
                  for (unsigned int qy = 0; qy < n_q_points_1d; ++qy)
                    {
                      const VectorizedArrayType z =
                        make_vectorized_array<VectorizedArrayType>(quad_1d.point(qz)[0]);
                      const VectorizedArrayType y =
                        make_vectorized_array<VectorizedArrayType>(quad_1d.point(qy)[0]);
                      // x-derivative, already complete
                      Tensor<1, dim, VectorizedArrayType> x_con = v[1] + z * v[5];
                      x_con += y * (v[4] + z * v[7]);
                      // y-derivative, constant part
                      Tensor<1, dim, VectorizedArrayType> y_con = v[2] + z * v[6];
                      // y-derivative, xi-dependent part
                      Tensor<1, dim, VectorizedArrayType> y_var = v[4] + z * v[7];
                      // z-derivative, constant part
                      Tensor<1, dim, VectorizedArrayType> z_con = v[3] + y * v[6];
                      // z-derivative, variable part
                      Tensor<1, dim, VectorizedArrayType> z_var = v[5] + y * v[7];
                      double q_weight_tmp = quad_1d.weight(qz) * quad_1d.weight(qy);
                      for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                        {
                          const VectorizedArrayType x =
                            make_vectorized_array<VectorizedArrayType>(quad_1d.point(qx)[0]);
                          Tensor<2, dim, VectorizedArrayType> jac;
                          jac[1] = y_con + x * y_var;
                          jac[2] = z_con + x * z_var;
                          for (unsigned int d = 0; d < dim; ++d)
                            jac[0][d] = x_con[d];
                          VectorizedArrayType det = do_invert(jac);
                          det                     = det * (q_weight_tmp * quad_1d.weight(qx));

                          for (unsigned int c = 0; c < n_components; ++c)
                            {
                              const unsigned int  offset = c * dim * n_q_points;
                              VectorizedArrayType tmp[dim];
                              for (unsigned int d = 0; d < dim; ++d)
                                {
                                  tmp[d] = jac[d][0] * phi_grads[q + offset];
                                  for (unsigned int e = 1; e < dim; ++e)
                                    tmp[d] += jac[d][e] * phi_grads[q + e * n_q_points + offset];
                                  tmp[d] *= det;
                                }
                              for (unsigned int d = 0; d < dim; ++d)
                                {
                                  phi_grads[q + d * n_q_points + offset] = jac[0][d] * tmp[0];
                                  for (unsigned int e = 1; e < dim; ++e)
                                    phi_grads[q + d * n_q_points + offset] += jac[e][d] * tmp[e];
                                }
                            }
                        }
                    }
                }
            }
          VectorizedArrayType scratch[Utilities::pow(fe_degree + 1, dim) * n_components];
          if (do_sum)
            for (unsigned int i = 0; i < Utilities::pow(fe_degree + 1, dim) * n_components; ++i)
              scratch[i] = phi.begin_dof_values()[i];

          phi.integrate(false, true);

          if (do_sum)
            {
              VectorizedArrayType local_sum = VectorizedArrayType();
              for (unsigned int i = 0; i < Utilities::pow(fe_degree + 1, dim) * n_components; ++i)
                local_sum += phi.begin_dof_values()[i] * scratch[i];

              for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
                accumulated_sum += local_sum[v];
            }

          if (fe_degree > 2)
            for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
              distribute_local_to_global_compressed<dim, fe_degree, Number>(
                ::internal::get_block(dst, bl),
                compressed_dof_indices,
                all_indices_uniform,
                cell,
                phi.begin_dof_values() + bl * Utilities::pow(fe_degree + 1, dim));
          else
            phi.distribute_local_to_global(dst);
        }
    }

    void
    local_apply_merged(const MatrixFree<dim, value_type, VectorizedArrayType> &data,
                       VectorType &                                            dst,
                       const VectorType &                                      src,
                       const std::pair<unsigned int, unsigned int> &           cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        data);
      constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(src);
          phi.evaluate(false, true);
          VectorizedArrayType *phi_grads = phi.begin_gradients();
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              if (dim == 2)
                {
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      const unsigned int  offset = c * dim * n_q_points;
                      VectorizedArrayType tmp    = phi_grads[q + offset];
                      phi_grads[q + offset] = merged_coefficients[cell * n_q_points + q][0] * tmp +
                                              merged_coefficients[cell * n_q_points + q][1] *
                                                phi_grads[q + n_q_points + offset];
                      phi_grads[q + n_q_points + offset] =
                        merged_coefficients[cell * n_q_points + q][1] * tmp +
                        merged_coefficients[cell * n_q_points + q][2] *
                          phi_grads[q + n_q_points + offset];
                    }
                }
              else if (dim == 3)
                {
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      const unsigned int  offset = c * dim * n_q_points;
                      VectorizedArrayType tmp0   = phi_grads[q + offset];
                      VectorizedArrayType tmp1   = phi_grads[q + n_q_points + offset];
                      phi_grads[q + offset] =
                        (merged_coefficients[cell * n_q_points + q][0] * tmp0 +
                         merged_coefficients[cell * n_q_points + q][1] * tmp1 +
                         merged_coefficients[cell * n_q_points + q][2] *
                           phi_grads[q + 2 * n_q_points + offset]);
                      phi_grads[q + n_q_points + offset] =
                        (merged_coefficients[cell * n_q_points + q][1] * tmp0 +
                         merged_coefficients[cell * n_q_points + q][3] * tmp1 +
                         merged_coefficients[cell * n_q_points + q][4] *
                           phi_grads[q + 2 * n_q_points + offset]);
                      phi_grads[q + 2 * n_q_points + offset] =
                        (merged_coefficients[cell * n_q_points + q][2] * tmp0 +
                         merged_coefficients[cell * n_q_points + q][4] * tmp1 +
                         merged_coefficients[cell * n_q_points + q][5] *
                           phi_grads[q + 2 * n_q_points + offset]);
                    }
                }
            }
          phi.integrate(false, true);
          phi.distribute_local_to_global(dst);
        }
    }

    void
    local_apply_construct_q(const MatrixFree<dim, value_type, VectorizedArrayType> &data,
                            VectorType &                                            dst,
                            const VectorType &                                      src,
                            const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        data);
      constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);
      VectorizedArrayType    jacobians_z[dim * n_q_points];
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          if (fe_degree > 2)
            for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
              read_dof_values_compressed<dim, fe_degree, value_type>(
                ::internal::get_block(src, bl),
                compressed_dof_indices,
                all_indices_uniform,
                cell,
                phi.begin_dof_values() + bl * Utilities::pow(fe_degree + 1, dim));
          else
            phi.read_dof_values(src);
          phi.evaluate(false, true);
          VectorizedArrayType *phi_grads = phi.begin_gradients();
          if (dim == 3)
            for (unsigned int d = 0; d < dim; ++d)
              dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                       dim,
                                                       n_q_points_1d,
                                                       n_q_points_1d,
                                                       VectorizedArrayType,
                                                       VectorizedArrayType>::
                template apply<2, true, false, 1>(
                  data.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                  quadrature_points.begin() + (cell * dim + d) * n_q_points,
                  jacobians_z + d * n_q_points);
          for (unsigned int q2 = 0, q = 0; q2 < (dim == 3 ? n_q_points_1d : 1); ++q2)
            {
              constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
              VectorizedArrayType    jacobians_y[dim * n_q_points_2d];
              for (unsigned int d = 0; d < dim; ++d)
                dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                         2,
                                                         n_q_points_1d,
                                                         n_q_points_1d,
                                                         VectorizedArrayType,
                                                         VectorizedArrayType>::
                  template apply<1, true, false, 1>(
                    data.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    quadrature_points.begin() + (cell * dim + d) * n_q_points + q2 * n_q_points_2d,
                    jacobians_y + d * n_q_points_2d);
              for (unsigned int q1 = 0; q1 < n_q_points_1d; ++q1)
                {
                  VectorizedArrayType jacobians_x[dim * n_q_points_1d];
                  for (unsigned int d = 0; d < dim; ++d)
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             1,
                                                             n_q_points_1d,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>::
                      template apply<0, true, false, 1>(
                        data.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        quadrature_points.begin() + (cell * dim + d) * n_q_points +
                          q2 * n_q_points_2d + q1 * n_q_points_1d,
                        jacobians_x + d * n_q_points_1d);
                  for (unsigned int q0 = 0; q0 < n_q_points_1d; ++q0, ++q)
                    {
                      Tensor<2, dim, VectorizedArrayType> jac;
                      for (unsigned int e = 0; e < dim; ++e)
                        jac[2][e] = jacobians_z[e * n_q_points + q];
                      for (unsigned int e = 0; e < dim; ++e)
                        jac[1][e] = jacobians_y[e * n_q_points_2d + q1 * n_q_points_1d + q0];
                      for (unsigned int e = 0; e < dim; ++e)
                        jac[0][e] = jacobians_x[e * n_q_points_1d + q0];
                      VectorizedArrayType det = do_invert(jac);
                      det                     = det * (data.get_quadrature().weight(q));

                      for (unsigned int c = 0; c < n_components; ++c)
                        {
                          const unsigned int  offset = c * dim * n_q_points;
                          VectorizedArrayType tmp[dim];
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              tmp[d] = jac[d][0] * phi_grads[q + offset];
                              for (unsigned int e = 1; e < dim; ++e)
                                tmp[d] += jac[d][e] * phi_grads[q + e * n_q_points + offset];
                              tmp[d] *= det;
                            }
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              phi_grads[q + d * n_q_points + offset] = jac[0][d] * tmp[0];
                              for (unsigned int e = 1; e < dim; ++e)
                                phi_grads[q + d * n_q_points + offset] += jac[e][d] * tmp[e];
                            }
                        }
                    }
                }
            }
          phi.integrate(false, true);
          if (fe_degree > 2)
            for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
              distribute_local_to_global_compressed<dim, fe_degree, Number>(
                ::internal::get_block(dst, bl),
                compressed_dof_indices,
                all_indices_uniform,
                cell,
                phi.begin_dof_values() + bl * Utilities::pow(fe_degree + 1, dim));
          else
            phi.distribute_local_to_global(dst);
        }
    }

    std::shared_ptr<const MatrixFree<dim, Number, VectorizedArrayType>> data;

    mutable Number accumulated_sum;

    Quadrature<1> quad_1d;
    // For local_apply_linear_geo:
    // A list containing the geometry in terms of the bilinear coefficients,
    // i.e. x_0, x_1-x_0, x_2-x_0, x_4-x_0, x_3-x_2-x_1+x_0, ...
    AlignedVector<
      std::array<Tensor<1, dim, VectorizedArrayType>, GeometryInfo<dim>::vertices_per_cell>>
      cell_vertex_coefficients;

    // For local_apply_merged: dim*(dim+1)/2 coefficients
    AlignedVector<Tensor<1, (dim * (dim + 1) / 2), VectorizedArrayType>> merged_coefficients;

    // for local_apply_construct_q:
    // Data in all quadrature points in collocated space
    AlignedVector<VectorizedArrayType> quadrature_points;

    std::vector<unsigned int>  compressed_dof_indices;
    std::vector<unsigned char> all_indices_uniform;
  };
} // namespace Poisson

#endif
