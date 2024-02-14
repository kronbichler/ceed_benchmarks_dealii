
#ifndef poisson_operator_h
#define poisson_operator_h


#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "solver_cg_optimized.h"
#include "vector_access_reduced.h"

namespace Mass
{
  using namespace dealii;



  // general mass matrix operator for block case
  template <int dim,
            int fe_degree,
            int n_q_points_1d            = fe_degree + 1,
            int n_components_            = 1,
            typename Number              = double,
            typename VectorizedArrayType = VectorizedArray<Number>>
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
    MassOperator()
    {}

    /**
     * Initialize function.
     */
    void
    initialize(std::shared_ptr<const MatrixFree<dim, Number>> data_)
    {
      this->data = data_;
    }

    /**
     * Initialize function.
     */
    void
    initialize_dof_vector(LinearAlgebra::distributed::BlockVector<Number> &vec) const
    {
      vec.reinit(dim);
      for (unsigned int d = 0; d < dim; ++d)
        data->initialize_dof_vector(vec.block(d));
      vec.collect_sizes();
    }

    /**
     * Matrix-vector multiplication.
     */
    void
    vmult(LinearAlgebra::distributed::BlockVector<Number>       &dst,
          const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
      this->data->cell_loop(&MassOperator::local_apply_cell, this, dst, src, true);
    }

    /**
     * Matrix-vector multiplication including the inner product (locally,
     * without communicating yet)
     */
    Number
    vmult_inner_product(LinearAlgebra::distributed::BlockVector<Number>       &dst,
                        const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
      accumulated_sum = 0;
      this->data->cell_loop(&MassOperator::local_apply_cell_inner_product, this, dst, src, true);
      return accumulated_sum;
    }

    /**
     * Transpose matrix-vector multiplication. Since the mass matrix is
     * symmetric, it does exactly the same as vmult().
     */
    void
    Tvmult(LinearAlgebra::distributed::BlockVector<Number>       &dst,
           const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
      vmult(dst, src);
    }

  private:
    /**
     * Cell contribution for the mass matrix
     */
    void
    local_apply_cell(const MatrixFree<dim, value_type, VectorizedArrayType> &data,
                     LinearAlgebra::distributed::BlockVector<Number>        &dst,
                     const LinearAlgebra::distributed::BlockVector<Number>  &src,
                     const std::pair<unsigned int, unsigned int>            &cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(src);
          phi.evaluate(EvaluationFlags::values);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_value(phi.get_value(q), q);
          phi.integrate(EvaluationFlags::values);
          phi.distribute_local_to_global(dst);
        }
    }

    /**
     * Like before but including the product between the source and
     * destination
     */
    void
    local_apply_cell_inner_product(const MatrixFree<dim, value_type, VectorizedArrayType> &data,
                                   LinearAlgebra::distributed::BlockVector<Number>        &dst,
                                   const LinearAlgebra::distributed::BlockVector<Number>  &src,
                                   const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType>
        phi_read(data);
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi_read.reinit(cell);
          phi_read.read_dof_values(src);
          phi.evaluate(EvaluationFlags::values);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_value(phi.get_value(q), q);
          phi.integrate(EvaluationFlags::values);
          VectorizedArrayType local_sum = VectorizedArrayType();
          for (unsigned int i = 0; i < Utilities::pow(fe_degree + 1, dim) * n_components; ++i)
            local_sum += phi.begin_dof_values()[i] * phi_read.begin_dof_values()[i];
          phi.distribute_local_to_global(dst);
          for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
            accumulated_sum += local_sum[v];
        }
    }

    std::shared_ptr<const MatrixFree<dim, Number, VectorizedArrayType>> data;
    mutable double                                                      accumulated_sum;
  };



  // partial specialization for scalar case with non-block vectors
  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            typename Number,
            typename VectorizedArrayType>
  class MassOperator<dim, fe_degree, n_q_points_1d, 1, Number, VectorizedArrayType>
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
    MassOperator()
    {}

    /**
     * Initialize function.
     */
    void
    initialize(std::shared_ptr<const MatrixFree<dim, Number, VectorizedArrayType>> data_,
               const AffineConstraints<double>                                    &constraints)
    {
      this->data = data_;


      if (fe_degree > 2)
        {
          compressed_dof_indices.resize(Utilities::pow(3, dim) * VectorizedArrayType::size() *
                                          data->n_cell_batches(),
                                        numbers::invalid_unsigned_int);
          all_indices_uniform.resize(Utilities::pow(3, dim) * data->n_cell_batches(), 1);
        }

      std::vector<types::global_dof_index> dof_indices(
        data->get_dof_handler().get_fe().dofs_per_cell);
      constexpr unsigned int    n_lanes = VectorizedArrayType::size();
      std::vector<unsigned int> renumber_lex =
        FETools::hierarchic_to_lexicographic_numbering<dim>(2);
      for (auto &i : renumber_lex)
        i *= n_lanes;

      for (unsigned int c = 0; c < data->n_cell_batches(); ++c)
        {
          constexpr unsigned int n_lanes = VectorizedArrayType::size();
          for (unsigned int l = 0; l < data->n_active_entries_per_cell_batch(c); ++l)
            {
              const typename DoFHandler<dim>::cell_iterator cell = data->get_cell_iterator(c, l);
              if (fe_degree > 2)
                {
                  cell->get_dof_indices(dof_indices);
                  const unsigned int n_components = cell->get_fe().n_components();
                  const unsigned int offset       = Utilities::pow(3, dim) * (n_lanes * c) + l;
                  const Utilities::MPI::Partitioner &part =
                    *data->get_dof_info().vector_partitioner;
                  unsigned int cc = 0, cf = 0;
                  for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell;
                       ++i, ++cc, cf += n_components)
                    {
                      if (!constraints.is_constrained(dof_indices[cf]))
                        compressed_dof_indices[offset + renumber_lex[cc]] =
                          part.global_to_local(dof_indices[cf]);
                      for (unsigned int c = 0; c < n_components; ++c)
                        AssertThrow(dof_indices[cf + c] == dof_indices[cf] + c,
                                    ExcNotImplemented());
                    }

                  for (unsigned int line = 0; line < GeometryInfo<dim>::lines_per_cell; ++line)
                    {
                      const unsigned int size = fe_degree - 1;
                      if (!constraints.is_constrained(dof_indices[cf]))
                        {
                          for (unsigned int i = 0; i < size; ++i)
                            for (unsigned int c = 0; c < n_components; ++c)
                              AssertThrow(dof_indices[cf + c] == dof_indices[cf] + c,
                                          ExcNotImplemented());
                          compressed_dof_indices[offset + renumber_lex[cc]] =
                            part.global_to_local(dof_indices[cf]);
                        }
                      ++cc;
                      cf += size * n_components;
                    }
                  for (unsigned int quad = 0; quad < GeometryInfo<dim>::quads_per_cell; ++quad)
                    {
                      const unsigned int size = (fe_degree - 1) * (fe_degree - 1);
                      if (!constraints.is_constrained(dof_indices[cf]))
                        {
                          // switch order x-z for y faces in 3D to lexicographic layout
                          if (dim == 3 && (quad == 2 || quad == 3))
                            for (unsigned int i1 = 0, i = 0; i1 < fe_degree - 1; ++i1)
                              for (unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                                for (unsigned int c = 0; c < n_components; ++c)
                                  AssertThrow(dof_indices[cf + c] == dof_indices[cf] + c,
                                              ExcNotImplemented());
                          else
                            for (unsigned int i = 0; i < size; ++i)
                              for (unsigned int c = 0; c < n_components; ++c)
                                AssertThrow(dof_indices[cf + c] == dof_indices[cf] + c,
                                            ExcNotImplemented());
                          compressed_dof_indices[offset + renumber_lex[cc]] =
                            part.global_to_local(dof_indices[cf]);
                        }
                      ++cc;
                      cf += size * n_components;
                    }
                  for (unsigned int hex = 0; hex < GeometryInfo<dim>::hexes_per_cell; ++hex)
                    {
                      const unsigned int size = (fe_degree - 1) * (fe_degree - 1) * (fe_degree - 1);
                      if (!constraints.is_constrained(dof_indices[cf]))
                        {
                          for (unsigned int i = 0; i < size; ++i)
                            for (unsigned int c = 0; c < n_components; ++c)
                              AssertThrow(dof_indices[cf + c] == dof_indices[cf] + c,
                                          ExcNotImplemented());
                          compressed_dof_indices[offset + renumber_lex[cc]] =
                            part.global_to_local(dof_indices[cf]);
                        }
                      ++cc;
                      cf += size * n_components;
                    }
                  AssertThrow(cc == Utilities::pow(3, dim),
                              ExcMessage("Expected 3^dim dofs, got " + std::to_string(cc)));
                  AssertThrow(cf == dof_indices.size(),
                              ExcMessage("Expected (fe_degree+1)^dim dofs, got " +
                                         std::to_string(cf)));
                }
            }
          if (fe_degree > 2)
            {
              for (unsigned int i = 0; i < Utilities::pow<unsigned int>(3, dim); ++i)
                for (unsigned int v = 0; v < n_lanes; ++v)
                  if (compressed_dof_indices[Utilities::pow(3, dim) * (n_lanes * c) + i * n_lanes +
                                             v] == numbers::invalid_unsigned_int)
                    all_indices_uniform[Utilities::pow(3, dim) * c + i] = 0;
            }
        }
    }

    /**
     * Initialize function.
     */
    void
    initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> &vec) const
    {
      data->initialize_dof_vector(vec);
    }

    /**
     * Matrix-vector multiplication.
     */
    void
    vmult(LinearAlgebra::distributed::Vector<Number>       &dst,
          const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      this->data->cell_loop(&MassOperator::local_apply_cell, this, dst, src, true);
    }

    void
    vmult(
      LinearAlgebra::distributed::Vector<Number>                        &dst,
      const LinearAlgebra::distributed::Vector<Number>                  &src,
      const std::function<void(const unsigned int, const unsigned int)> &operation_before_loop,
      const std::function<void(const unsigned int, const unsigned int)> &operation_after_loop) const
    {
      this->data->cell_loop(&MassOperator::local_apply_cell,
                            this,
                            dst,
                            src,
                            operation_before_loop,
                            operation_after_loop);
    }

    void
    vmult_add(LinearAlgebra::distributed::Vector<Number>       &dst,
              const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      this->data->cell_loop(&MassOperator::local_apply_cell, this, dst, src, false);
    }

    /**
     * Matrix-vector multiplication including the inner product (locally,
     * without communicating yet)
     */
    Number
    vmult_inner_product(LinearAlgebra::distributed::Vector<Number>       &dst,
                        const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      accumulated_sum = 0;
      this->data->cell_loop(&MassOperator::local_apply_cell_inner_product, this, dst, src, true);
      return accumulated_sum;
    }

    Tensor<1, 7>
    vmult_with_merged_sums(LinearAlgebra::distributed::Vector<Number>                       &x,
                           LinearAlgebra::distributed::Vector<Number>                       &g,
                           LinearAlgebra::distributed::Vector<Number>                       &d,
                           LinearAlgebra::distributed::Vector<Number>                       &h,
                           const DiagonalMatrix<LinearAlgebra::distributed::Vector<Number>> &prec,
                           const Number                                                      alpha,
                           const Number                                                      beta,
                           const Number alpha_old,
                           const Number beta_old) const
    {
      Tensor<1, 7, VectorizedArray<Number>> sums;
      this->data->cell_loop(
        &MassOperator::local_apply_cell,
        this,
        h,
        d,
        [&](const unsigned int start_range, const unsigned int end_range) {
          do_cg_update4b<1, Number, true>(start_range,
                                          end_range,
                                          h.begin(),
                                          x.begin(),
                                          g.begin(),
                                          d.begin(),
                                          prec.get_vector().begin(),
                                          alpha,
                                          beta,
                                          alpha_old,
                                          beta_old);
        },
        [&](const unsigned int start_range, const unsigned int end_range) {
          do_cg_update3b<1, Number>(start_range,
                                    end_range,
                                    g.begin(),
                                    d.begin(),
                                    h.begin(),
                                    prec.get_vector().begin(),
                                    sums);
        });

      dealii::Tensor<1, 7> results;
      for (unsigned int i = 0; i < 7; ++i)
        {
          results[i] = sums[i][0];
          for (unsigned int v = 1; v < VectorizedArrayType::size(); ++v)
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
    void
    Tvmult(LinearAlgebra::distributed::Vector<Number>       &dst,
           const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      vmult(dst, src);
    }

  private:
    /**
     * For this operator, there is just a cell contribution.
     */
    void
    local_apply_cell(const MatrixFree<dim, value_type, VectorizedArrayType> &data,
                     LinearAlgebra::distributed::Vector<Number>             &dst,
                     const LinearAlgebra::distributed::Vector<Number>       &src,
                     const std::pair<unsigned int, unsigned int>            &cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number, VectorizedArrayType> phi(data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          if (fe_degree > 2)
            read_dof_values_compressed<dim, fe_degree, fe_degree + 1, 1, value_type>(
              src,
              compressed_dof_indices,
              all_indices_uniform,
              cell,
              {},
              true,
              phi.begin_dof_values());
          else
            phi.read_dof_values(src);

          phi.evaluate(EvaluationFlags::values);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_value(phi.get_value(q), q);
          phi.integrate(EvaluationFlags::values);

          if (fe_degree > 2)
            distribute_local_to_global_compressed<dim, fe_degree, fe_degree + 1, 1, value_type>(
              dst,
              compressed_dof_indices,
              all_indices_uniform,
              cell,
              {},
              true,
              phi.begin_dof_values());
          else
            phi.distribute_local_to_global(dst);
        }
    }

    /**
     * Like before but including the product between the source and
     * destination
     */
    void
    local_apply_cell_inner_product(const MatrixFree<dim, value_type, VectorizedArrayType> &data,
                                   LinearAlgebra::distributed::Vector<Number>             &dst,
                                   const LinearAlgebra::distributed::Vector<Number>       &src,
                                   const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number, VectorizedArrayType> phi(data);
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number, VectorizedArrayType> phi_read(data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi_read.reinit(cell);
          phi_read.read_dof_values(src);
          phi.evaluate(phi_read.begin_dof_values(), EvaluationFlags::values);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_value(phi.get_value(q), q);
          phi.integrate(EvaluationFlags::values);
          VectorizedArrayType local_sum = VectorizedArrayType();
          for (unsigned int i = 0; i < Utilities::pow<unsigned int>(fe_degree + 1, dim); ++i)
            local_sum += phi.begin_dof_values()[i] * phi_read.begin_dof_values()[i];
          phi.distribute_local_to_global(dst);
          for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
            accumulated_sum += local_sum[v];
        }
    }

    std::shared_ptr<const MatrixFree<dim, Number, VectorizedArrayType>> data;
    mutable Number                                                      accumulated_sum;

    std::vector<unsigned int>  compressed_dof_indices;
    std::vector<unsigned char> all_indices_uniform;
  };

} // namespace Mass


#endif
