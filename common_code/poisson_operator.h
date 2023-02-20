
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
  inline DEAL_II_ALWAYS_INLINE Number
  do_invert(Tensor<2, 2, Number> &t)
  {
    const Number det     = t[0][0] * t[1][1] - t[1][0] * t[0][1];
    const Number inv_det = 1.0 / det;
    const Number tmp     = inv_det * t[0][0];
    t[0][0]              = inv_det * t[1][1];
    t[0][1]              = -inv_det * t[0][1];
    t[1][0]              = -inv_det * t[1][0];
    t[1][1]              = tmp;
    return det;
  }


  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE Number
  do_invert(Tensor<2, 3, Number> &t)
  {
    const Number tr00    = t[1][1] * t[2][2] - t[1][2] * t[2][1];
    const Number tr10    = t[1][2] * t[2][0] - t[1][0] * t[2][2];
    const Number tr20    = t[1][0] * t[2][1] - t[1][1] * t[2][0];
    const Number det     = t[0][0] * tr00 + t[0][1] * tr10 + t[0][2] * tr20;
    const Number inv_det = 1.0 / det;
    const Number tr01    = t[0][2] * t[2][1] - t[0][1] * t[2][2];
    const Number tr02    = t[0][1] * t[1][2] - t[0][2] * t[1][1];
    const Number tr11    = t[0][0] * t[2][2] - t[0][2] * t[2][0];
    const Number tr12    = t[0][2] * t[1][0] - t[0][0] * t[1][2];
    t[2][1]              = inv_det * (t[0][1] * t[2][0] - t[0][0] * t[2][1]);
    t[2][2]              = inv_det * (t[0][0] * t[1][1] - t[0][1] * t[1][0]);
    t[0][0]              = inv_det * tr00;
    t[0][1]              = inv_det * tr01;
    t[0][2]              = inv_det * tr02;
    t[1][0]              = inv_det * tr10;
    t[1][1]              = inv_det * tr11;
    t[1][2]              = inv_det * tr12;
    t[2][0]              = inv_det * tr20;
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



  // expand templated operations
#define EXPAND_OPERATIONS(OPERATION)                                                    \
  const unsigned int degree        = data->get_dof_handler().get_fe().degree;           \
  const unsigned int n_q_points_1d = data->get_shape_info().data[0].n_q_points_1d;      \
                                                                                        \
  if (degree == 1 && n_q_points_1d == 2)                                                \
    OPERATION<1, 2>(dst, src, cell_range);                                              \
  else if (degree == 1 && n_q_points_1d == 3)                                           \
    OPERATION<1, 3>(dst, src, cell_range);                                              \
  else if (degree == 2 && n_q_points_1d == 3)                                           \
    OPERATION<2, 3>(dst, src, cell_range);                                              \
  else if (degree == 2 && n_q_points_1d == 4)                                           \
    OPERATION<2, 4>(dst, src, cell_range);                                              \
  else if (degree == 3 && n_q_points_1d == 4)                                           \
    OPERATION<3, 4>(dst, src, cell_range);                                              \
  else if (degree == 3 && n_q_points_1d == 5)                                           \
    OPERATION<3, 5>(dst, src, cell_range);                                              \
  else if (degree == 4 && n_q_points_1d == 5)                                           \
    OPERATION<4, 5>(dst, src, cell_range);                                              \
  else if (degree == 4 && n_q_points_1d == 6)                                           \
    OPERATION<4, 6>(dst, src, cell_range);                                              \
  else if (degree == 5 && n_q_points_1d == 6)                                           \
    OPERATION<5, 6>(dst, src, cell_range);                                              \
  else if (degree == 5 && n_q_points_1d == 7)                                           \
    OPERATION<5, 7>(dst, src, cell_range);                                              \
  else if (degree == 6 && n_q_points_1d == 7)                                           \
    OPERATION<6, 7>(dst, src, cell_range);                                              \
  else if (degree == 6 && n_q_points_1d == 8)                                           \
    OPERATION<6, 8>(dst, src, cell_range);                                              \
  else if (degree == 7 && n_q_points_1d == 8)                                           \
    OPERATION<7, 8>(dst, src, cell_range);                                              \
  else if (degree == 7 && n_q_points_1d == 9)                                           \
    OPERATION<7, 9>(dst, src, cell_range);                                              \
  else if (degree == 8 && n_q_points_1d == 9)                                           \
    OPERATION<8, 9>(dst, src, cell_range);                                              \
  else if (degree == 8 && n_q_points_1d == 10)                                          \
    OPERATION<8, 10>(dst, src, cell_range);                                             \
  else if (degree == 9 && n_q_points_1d == 10)                                          \
    OPERATION<9, 10>(dst, src, cell_range);                                             \
  else if (degree == 9 && n_q_points_1d == 11)                                          \
    OPERATION<9, 11>(dst, src, cell_range);                                             \
  else if (degree == 10 && n_q_points_1d == 11)                                         \
    OPERATION<10, 11>(dst, src, cell_range);                                            \
  else if (degree == 10 && n_q_points_1d == 12)                                         \
    OPERATION<10, 12>(dst, src, cell_range);                                            \
  else if (degree == 11 && n_q_points_1d == 12)                                         \
    OPERATION<11, 12>(dst, src, cell_range);                                            \
  else if (degree == 11 && n_q_points_1d == 13)                                         \
    OPERATION<11, 13>(dst, src, cell_range);                                            \
  else                                                                                  \
    AssertThrow(false,                                                                  \
                ExcNotImplemented("Degree " + std::to_string(degree) + " n_q_points " + \
                                  std::to_string(n_q_points_1d) + " not implemented"));



  // similar to bp1/bench_mass.cc but now implementing the Laplacian in various
  // options.

  template <int dim,
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
      const unsigned int fe_degree     = data_->get_dof_handler(0).get_fe().degree;
      const unsigned int n_q_points_1d = data_->get_shape_info().data[0].n_q_points_1d;
      fast_read  = !data_->get_dof_handler(0).get_triangulation().has_hanging_nodes();
      this->data = data_;
      quad_1d    = QGauss<1>(n_q_points_1d);
      cell_vertex_coefficients.resize(data->n_cell_batches());
      cell_quadratic_coefficients.resize(data->n_cell_batches());
      const std::size_t n_q_points = Utilities::fixed_power<dim>(quad_1d.size());
      merged_coefficients.resize(n_q_points * data->n_cell_batches());
      quadrature_points.resize(n_q_points * dim * data->n_cell_batches());

      coefficients_eo.resize(data_->get_shape_info().data[0].shape_gradients_collocation_eo.size());
      for (unsigned int i = 0; i < coefficients_eo.size(); ++i)
        coefficients_eo[i] = data_->get_shape_info().data[0].shape_gradients_collocation_eo[i][0];

      if (false && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          const auto &              di = data->get_dof_info(0);
          const auto &              ti = data->get_task_info();
          std::vector<unsigned int> distances(di.vector_partitioner->locally_owned_size() / 64 + 1,
                                              numbers::invalid_unsigned_int);
          for (unsigned int id =
                 di.cell_loop_pre_list_index[ti.partition_row_index[ti.partition_row_index.size() -
                                                                    2]];
               id < di.cell_loop_pre_list_index
                      [ti.partition_row_index[ti.partition_row_index.size() - 2] + 1];
               ++id)
            for (unsigned int a = di.cell_loop_pre_list[id].first;
                 a < di.cell_loop_pre_list[id].second;
                 a += 64)
              distances[a / 64] = 0;

          for (unsigned int part = 0; part < ti.partition_row_index.size() - 2; ++part)
            {
              for (unsigned int i = ti.partition_row_index[part];
                   i < ti.partition_row_index[part + 1];
                   ++i)
                {
                  // std::cout << "pre ";
                  for (unsigned int id = di.cell_loop_pre_list_index[i];
                       id != di.cell_loop_pre_list_index[i + 1];
                       ++id)
                    {
                      for (unsigned int a = di.cell_loop_pre_list[id].first;
                           a < di.cell_loop_pre_list[id].second;
                           a += 64)
                        distances[a / 64] = i;
                      // std::cout << id << "[" << di.cell_loop_pre_list[id].first << ","
                      //          << di.cell_loop_pre_list[id].second << ") ";
                    }
                  // std::cout << std::endl;
                  // std::cout << "post ";
                  for (unsigned int id = di.cell_loop_post_list_index[i];
                       id != di.cell_loop_post_list_index[i + 1];
                       ++id)
                    {
                      for (unsigned int a = di.cell_loop_post_list[id].first;
                           a < di.cell_loop_post_list[id].second;
                           a += 64)
                        distances[a / 64] = i - distances[a / 64];
                      // std::cout << id << "[" << di.cell_loop_post_list[id].first << ","
                      //          << di.cell_loop_post_list[id].second << ") ";
                    }
                  // std::cout << std::endl;
                }
              // std::cout << std::endl;
            }
          for (unsigned int id =
                 di.cell_loop_post_list_index[ti.partition_row_index[ti.partition_row_index.size() -
                                                                     2]];
               id < di.cell_loop_post_list_index
                      [ti.partition_row_index[ti.partition_row_index.size() - 2] + 1];
               ++id)
            for (unsigned int a = di.cell_loop_post_list[id].first;
                 a < di.cell_loop_post_list[id].second;
                 a += 64)
              distances[a / 64] =
                ti.partition_row_index[ti.partition_row_index.size() - 2] - distances[a / 64];
          std::map<unsigned int, unsigned int> count;
          for (const auto a : distances)
            count[a]++;
          for (const auto a : count)
            std::cout << a.first << " " << a.second << "   ";
          std::cout << std::endl;
        }

      if (fe_degree > 2)
        {
          compressed_dof_indices.resize(Utilities::pow(3, dim) * VectorizedArrayType::size() *
                                          data->n_cell_batches(),
                                        numbers::invalid_unsigned_int);
          all_indices_uniform.resize(Utilities::pow(3, dim) * data->n_cell_batches(), 1);
        }

      FE_Nothing<dim>                      dummy_fe;
      FEValues<dim>                        fe_values(dummy_fe,
                              Quadrature<dim>(quad_1d),
                              update_quadrature_points | update_jacobians | update_JxW_values);
      std::vector<types::global_dof_index> dof_indices(
        data->get_dof_handler().get_fe().dofs_per_cell);

      constexpr unsigned int    n_lanes = VectorizedArrayType::size();
      std::vector<unsigned int> renumber_lex =
        FETools::hierarchic_to_lexicographic_numbering<dim>(2);
      for (auto &i : renumber_lex)
        i *= n_lanes;

      for (unsigned int c = 0; c < data->n_cell_batches(); ++c)
        {
          for (unsigned int l = 0; l < data->n_active_entries_per_cell_batch(c); ++l)
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

                      // for now use only constant and linear term for the
                      // quadratic approximation
                      cell_quadratic_coefficients[c][0][d][l] =
                        cell_vertex_coefficients[c][0][d][l];
                      cell_quadratic_coefficients[c][1][d][l] =
                        cell_vertex_coefficients[c][1][d][l];
                      cell_quadratic_coefficients[c][3][d][l] =
                        cell_vertex_coefficients[c][2][d][l];
                      cell_quadratic_coefficients[c][4][d][l] =
                        cell_vertex_coefficients[c][3][d][l];
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

                      // for now use only constant and linear term for the
                      // quadratic approximation
                      cell_quadratic_coefficients[c][0][d][l] =
                        cell_vertex_coefficients[c][0][d][l];
                      cell_quadratic_coefficients[c][1][d][l] =
                        cell_vertex_coefficients[c][1][d][l];
                      cell_quadratic_coefficients[c][3][d][l] =
                        cell_vertex_coefficients[c][2][d][l];
                      cell_quadratic_coefficients[c][4][d][l] =
                        cell_vertex_coefficients[c][4][d][l];
                      cell_quadratic_coefficients[c][9][d][l] =
                        cell_vertex_coefficients[c][3][d][l];
                      cell_quadratic_coefficients[c][10][d][l] =
                        cell_vertex_coefficients[c][5][d][l];
                      cell_quadratic_coefficients[c][12][d][l] =
                        cell_vertex_coefficients[c][6][d][l];
                      cell_quadratic_coefficients[c][13][d][l] =
                        cell_vertex_coefficients[c][7][d][l];
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
                                    ExcMessage("Expected contiguous numbering"));
                    }

                  for (unsigned int line = 0; line < GeometryInfo<dim>::lines_per_cell; ++line)
                    {
                      const unsigned int size = fe_degree - 1;
                      if (!constraints.is_constrained(dof_indices[cf]))
                        {
                          for (unsigned int i = 0; i < size; ++i)
                            for (unsigned int c = 0; c < n_components; ++c)
                              AssertThrow(dof_indices[cf + c * size + i] ==
                                            dof_indices[cf] + i * n_components + c,
                                          ExcMessage("Expected contiguous numbering"));
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
                                  {
                                    AssertThrow(
                                      dof_indices[cf + c * size + i0 * (fe_degree - 1) + i1] ==
                                        dof_indices[cf] + i * n_components + c,
                                      ExcMessage("Expected contiguous numbering"));
                                  }
                          else
                            for (unsigned int i = 0; i < size; ++i)
                              for (unsigned int c = 0; c < n_components; ++c)
                                AssertThrow(dof_indices[cf + c * size + i] ==
                                              dof_indices[cf] + i * n_components + c,
                                            ExcMessage("Expected contiguous numbering"));
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
                              AssertThrow(dof_indices[cf + c * size + i] ==
                                            dof_indices[cf] + i * n_components + c,
                                          ExcMessage("Expected contiguous numbering"));
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
          // insert dummy entries to prevent geometry from degeneration and
          // subsequent division by zero, assuming a Cartesian geometry
          for (unsigned int l = data->n_active_entries_per_cell_batch(c);
               l < VectorizedArrayType::size();
               ++l)
            {
              for (unsigned int d = 0; d < dim; ++d)
                cell_vertex_coefficients[c][d + 1][d][l] = 1.;
              cell_quadratic_coefficients[c][1][0][l] = 1.;
              if (dim > 1)
                cell_quadratic_coefficients[c][3][1][l] = 1.;
              if (dim > 2)
                cell_quadratic_coefficients[c][9][2][l] = 1.;
            }

          if (fe_degree > 2)
            {
              for (unsigned int i = 0; i < Utilities::pow<unsigned int>(3, dim); ++i)
                for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
                  if (compressed_dof_indices[Utilities::pow<unsigned int>(3, dim) *
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
      this->data->cell_loop(&LaplaceOperator::local_apply_quadratic_geo, this, dst, src, true);
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
      this->data->cell_loop(
        &LaplaceOperator::template local_apply_linear_geo<false>,
        this,
        h,
        d,
        [&](const unsigned int start_range, const unsigned int end_range) {
          for (unsigned int bl = 0; bl < ::internal::get_n_blocks(x); ++bl)
            do_cg_update4b<1, Number, true>(start_range,
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
      constexpr unsigned int n_read_components =
        IsBlockVector<VectorType>::value ? 1 : n_components;
      Tensor<1, 7, VectorizedArray<Number>> sums;
      this->data->cell_loop(
        &LaplaceOperator::local_apply_quadratic_geo,
        this,
        h,
        d,
        [&](const unsigned int start_range, const unsigned int end_range) {
          for (unsigned int bl = 0; bl < ::internal::get_n_blocks(x); ++bl)
            do_cg_update4b<n_read_components, Number, true>(start_range,
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
            do_cg_update3b<n_read_components, Number>(start_range,
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
     * Matrix-vector multiplication.
     */
    void
    vmult_quadratic(VectorType &dst, const VectorType &src) const
    {
      this->data->cell_loop(
        &LaplaceOperator::template local_apply_linear_geo<false>, this, dst, src, true);
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

    void
    apply_local(AlignedVector<VectorizedArrayType> &      dst,
                const AlignedVector<VectorizedArrayType> &src) const
    {
      const std::pair<unsigned int, unsigned int> cell_range(0, data->n_cell_batches());
      EXPAND_OPERATIONS(apply_local_impl);
    }

    template <int fe_degree, int n_q_points_1d>
    void
    apply_local_impl(AlignedVector<VectorizedArrayType> &      dst,
                     const AlignedVector<VectorizedArrayType> &src,
                     const std::pair<unsigned int, unsigned int> &) const
    {
      AssertThrow(n_q_points_1d == fe_degree + 1, ExcNotImplemented());
      constexpr unsigned int n_q_points    = Utilities::pow(n_q_points_1d, dim);
      constexpr unsigned int n_q_points_2d = Utilities::pow(n_q_points_1d, 2);

      using TensorType = Tensor<1, dim, VectorizedArrayType>;
      using Eval = dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                            dim,
                                                            n_q_points_1d,
                                                            n_q_points_1d,
                                                            VectorizedArrayType,
                                                            VectorizedArrayType>;

      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        *data);

      for (unsigned int cell = 0; cell < data->n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          const VectorizedArrayType *src_ptr   = src.data() + cell * n_q_points * n_components;
          VectorizedArrayType *      dst_ptr   = dst.data() + cell * n_q_points * n_components;
          VectorizedArrayType *      phi_grads = phi.begin_gradients();
          if (dim > 2)
            {
              for (unsigned int c = 0; c < n_components; ++c)
                Eval::template apply<2, true, false, 1>(
                  phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                  src_ptr + c * n_q_points,
                  phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points);
            }
          for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
            {
              using Eval2 =
                dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                         2,
                                                         n_q_points_1d,
                                                         n_q_points_1d,
                                                         VectorizedArrayType,
                                                         VectorizedArrayType>;
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval2::template apply<1, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    src_ptr + c * n_q_points + qz * n_q_points_2d,
                    phi_grads + (2 * c + 1) * n_q_points_2d);
                  Eval2::template apply<0, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    src_ptr + c * n_q_points + qz * n_q_points_2d,
                    phi_grads + 2 * c * n_q_points_2d);
                }
              for (unsigned int qxy = 0; qxy < n_q_points_2d; ++qxy, ++q)
                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    VectorizedArrayType tmp0 = phi_grads[qxy + c * 2 * n_q_points_2d];
                    VectorizedArrayType tmp1 = phi_grads[qxy + (c * 2 + 1) * n_q_points_2d];
                    phi_grads[qxy + c * 2 * n_q_points_2d] =
                      (merged_coefficients[cell * n_q_points + q][0] * tmp0 +
                       merged_coefficients[cell * n_q_points + q][1] * tmp1 +
                       merged_coefficients[cell * n_q_points + q][2] *
                         phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points]);
                    phi_grads[qxy + (c * 2 + 1) * n_q_points_2d] =
                      (merged_coefficients[cell * n_q_points + q][1] * tmp0 +
                       merged_coefficients[cell * n_q_points + q][3] * tmp1 +
                       merged_coefficients[cell * n_q_points + q][4] *
                         phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points]);
                    phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] =
                      (merged_coefficients[cell * n_q_points + q][2] * tmp0 +
                       merged_coefficients[cell * n_q_points + q][4] * tmp1 +
                       merged_coefficients[cell * n_q_points + q][5] *
                         phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points]);
                  }
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval2::template apply<0, false, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + 2 * c * n_q_points_2d,
                    dst_ptr + c * n_q_points + qz * n_q_points_2d);
                  Eval2::template apply<1, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + (2 * c + 1) * n_q_points_2d,
                    dst_ptr + c * n_q_points + qz * n_q_points_2d);
                }
            }
          for (unsigned int c = 0; c < n_components; ++c)
            {
              Eval::template apply<2, false, true, 1>(
                phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points,
                dst_ptr + c * n_q_points);
            }
        }
    }

    void
    apply_basic(AlignedVector<VectorizedArrayType> &      dst,
                const AlignedVector<VectorizedArrayType> &src) const
    {
      const std::pair<unsigned int, unsigned int> cell_range(0, data->n_cell_batches());
      EXPAND_OPERATIONS(apply_basic_impl);
    }

    template <int fe_degree, int n_q_points_1d>
    void
    apply_basic_impl(AlignedVector<VectorizedArrayType> &      dst,
                     const AlignedVector<VectorizedArrayType> &src,
                     const std::pair<unsigned int, unsigned int> &) const
    {
      AssertThrow(n_q_points_1d == fe_degree + 1, ExcNotImplemented());
      constexpr unsigned int n_q_points    = Utilities::pow(n_q_points_1d, dim);
      constexpr unsigned int n_q_points_2d = Utilities::pow(n_q_points_1d, 2);

      using TensorType = Tensor<1, dim, VectorizedArrayType>;
      using Eval = dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                            dim,
                                                            n_q_points_1d,
                                                            n_q_points_1d,
                                                            VectorizedArrayType,
                                                            VectorizedArrayType>;

      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        *data);

      for (unsigned int cell = 0; cell < data->n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          const VectorizedArrayType *src_ptr   = src.data() + cell * n_q_points * n_components;
          VectorizedArrayType *      dst_ptr   = dst.data() + cell * n_q_points * n_components;
          VectorizedArrayType *      phi_grads = phi.begin_gradients();
          if (dim > 2)
            {
              for (unsigned int c = 0; c < n_components; ++c)
                Eval::template apply<2, true, false, 1>(
                  phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                  src_ptr + c * n_q_points,
                  phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points);
            }
          for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
            {
              using Eval2 =
                dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                         2,
                                                         n_q_points_1d,
                                                         n_q_points_1d,
                                                         VectorizedArrayType,
                                                         VectorizedArrayType>;
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval2::template apply<1, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    src_ptr + c * n_q_points + qz * n_q_points_2d,
                    phi_grads + (2 * c + 1) * n_q_points_2d);
                  Eval2::template apply<0, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    src_ptr + c * n_q_points + qz * n_q_points_2d,
                    phi_grads + 2 * c * n_q_points_2d);
                }
              for (unsigned int qxy = 0; qxy < n_q_points_2d; ++qxy, ++q)
                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    const VectorizedArrayType det = phi.JxW(q);
                    const auto                jac = phi.inverse_jacobian(q);

                    VectorizedArrayType tmp[dim], tmp2[dim];
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        tmp[d] = jac[d][0] * phi_grads[qxy + c * 2 * n_q_points_2d] +
                                 jac[d][1] * phi_grads[qxy + (c * 2 + 1) * n_q_points_2d] +
                                 jac[d][2] *
                                   phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points];
                        tmp[d] *= det;
                      }
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        tmp2[d] = jac[0][d] * tmp[0];
                        for (unsigned int e = 1; e < dim; ++e)
                          tmp2[d] += jac[e][d] * tmp[e];
                      }
                    phi_grads[qxy + c * 2 * n_q_points_2d]                           = tmp2[0];
                    phi_grads[qxy + (c * 2 + 1) * n_q_points_2d]                     = tmp2[1];
                    phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] = tmp2[2];
                  }
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval2::template apply<0, false, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + 2 * c * n_q_points_2d,
                    dst_ptr + c * n_q_points + qz * n_q_points_2d);
                  Eval2::template apply<1, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + (2 * c + 1) * n_q_points_2d,
                    dst_ptr + c * n_q_points + qz * n_q_points_2d);
                }
            }
          for (unsigned int c = 0; c < n_components; ++c)
            {
              Eval::template apply<2, false, true, 1>(
                phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points,
                dst_ptr + c * n_q_points);
            }
        }
    }

    void
    apply_q(AlignedVector<VectorizedArrayType> &      dst,
            const AlignedVector<VectorizedArrayType> &src) const
    {
      const std::pair<unsigned int, unsigned int> cell_range(0, data->n_cell_batches());
      EXPAND_OPERATIONS(apply_q_impl);
    }

    template <int fe_degree, int n_q_points_1d>
    void
    apply_q_impl(AlignedVector<VectorizedArrayType> &      dst,
                 const AlignedVector<VectorizedArrayType> &src,
                 const std::pair<unsigned int, unsigned int> &) const
    {
      AssertThrow(n_q_points_1d == fe_degree + 1, ExcNotImplemented());
      constexpr unsigned int n_q_points    = Utilities::pow(n_q_points_1d, dim);
      constexpr unsigned int n_q_points_2d = Utilities::pow(n_q_points_1d, 2);

      using TensorType = Tensor<1, dim, VectorizedArrayType>;
      using Eval = dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                            dim,
                                                            n_q_points_1d,
                                                            n_q_points_1d,
                                                            VectorizedArrayType,
                                                            VectorizedArrayType>;

      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        *data);
      VectorizedArrayType jacobians_z[dim * n_q_points];

      for (unsigned int cell = 0; cell < data->n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          const VectorizedArrayType *src_ptr   = src.data() + cell * n_q_points * n_components;
          VectorizedArrayType *      dst_ptr   = dst.data() + cell * n_q_points * n_components;
          VectorizedArrayType *      phi_grads = phi.begin_gradients();
          if (dim > 2)
            {
              for (unsigned int c = 0; c < n_components; ++c)
                Eval::template apply<2, true, false, 1>(
                  phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                  src_ptr + c * n_q_points,
                  phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points);

              for (unsigned int d = 0; d < dim; ++d)
                dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                         dim,
                                                         n_q_points_1d,
                                                         n_q_points_1d,
                                                         VectorizedArrayType,
                                                         VectorizedArrayType>::
                  template apply<2, true, false, 1>(
                    data->get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    quadrature_points.begin() + (cell * dim + d) * n_q_points,
                    jacobians_z + d * n_q_points);
            }
          for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
            {
              using Eval2 =
                dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                         2,
                                                         n_q_points_1d,
                                                         n_q_points_1d,
                                                         VectorizedArrayType,
                                                         VectorizedArrayType>;
              VectorizedArrayType jacobians_y[dim * n_q_points_2d];
              for (unsigned int d = 0; d < dim; ++d)
                Eval2::template apply<1, true, false, 1>(
                  data->get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                  quadrature_points.begin() + (cell * dim + d) * n_q_points + qz * n_q_points_2d,
                  jacobians_y + d * n_q_points_2d);

              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval2::template apply<1, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    src_ptr + c * n_q_points + qz * n_q_points_2d,
                    phi_grads + (2 * c + 1) * n_q_points_2d);
                  Eval2::template apply<0, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    src_ptr + c * n_q_points + qz * n_q_points_2d,
                    phi_grads + 2 * c * n_q_points_2d);
                }

              for (unsigned int qy = 0; qy < n_q_points_1d; ++qy)
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
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        quadrature_points.begin() + (cell * dim + d) * n_q_points +
                          qz * n_q_points_2d + qy * n_q_points_1d,
                        jacobians_x + d * n_q_points_1d);
                  for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                    {
                      Tensor<2, dim, VectorizedArrayType> jac;
                      for (unsigned int e = 0; e < dim; ++e)
                        jac[2][e] = jacobians_z[e * n_q_points + q];
                      for (unsigned int e = 0; e < dim; ++e)
                        jac[1][e] = jacobians_y[e * n_q_points_2d + qy * n_q_points_1d + qx];
                      for (unsigned int e = 0; e < dim; ++e)
                        jac[0][e] = jacobians_x[e * n_q_points_1d + qx];
                      VectorizedArrayType det = do_invert(jac);
                      det                     = det * (data->get_quadrature().weight(q));

                      for (unsigned int c = 0; c < n_components; ++c)
                        {
                          const VectorizedArrayType det = phi.JxW(q);
                          const auto                jac = phi.inverse_jacobian(q);

                          VectorizedArrayType tmp[dim], tmp2[dim];
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              tmp[d] =
                                jac[d][0] *
                                  phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] +
                                jac[d][1] *
                                  phi_grads[qy * n_q_points_1d + qx + (c * 2 + 1) * n_q_points_2d] +
                                jac[d][2] *
                                  phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points];
                              tmp[d] *= det;
                            }
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              tmp2[d] = jac[0][d] * tmp[0];
                              for (unsigned int e = 1; e < dim; ++e)
                                tmp2[d] += jac[e][d] * tmp[e];
                            }
                          phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] = tmp2[0];
                          phi_grads[qy * n_q_points_1d + qx + (c * 2 + 1) * n_q_points_2d] =
                            tmp2[1];
                          phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] =
                            tmp2[2];
                        }
                    }
                }
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval2::template apply<0, false, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + 2 * c * n_q_points_2d,
                    dst_ptr + c * n_q_points + qz * n_q_points_2d);
                  Eval2::template apply<1, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + (2 * c + 1) * n_q_points_2d,
                    dst_ptr + c * n_q_points + qz * n_q_points_2d);
                }
            }
          for (unsigned int c = 0; c < n_components; ++c)
            {
              Eval::template apply<2, false, true, 1>(
                phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points,
                dst_ptr + c * n_q_points);
            }
        }
    }

    void
    apply_q1(AlignedVector<VectorizedArrayType> &      dst,
             const AlignedVector<VectorizedArrayType> &src) const
    {
      const std::pair<unsigned int, unsigned int> cell_range(0, data->n_cell_batches());
      EXPAND_OPERATIONS(apply_q1_impl);
    }

    template <int fe_degree, int n_q_points_1d>
    void
    apply_q1_impl(AlignedVector<VectorizedArrayType> &      dst,
                  const AlignedVector<VectorizedArrayType> &src,
                  const std::pair<unsigned int, unsigned int> &) const
    {
      AssertThrow(n_q_points_1d == fe_degree + 1, ExcNotImplemented());
      constexpr unsigned int n_q_points    = Utilities::pow(n_q_points_1d, dim);
      constexpr unsigned int n_q_points_2d = Utilities::pow(n_q_points_1d, 2);

      using TensorType = Tensor<1, dim, VectorizedArrayType>;
      using Eval = dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                            dim,
                                                            n_q_points_1d,
                                                            n_q_points_1d,
                                                            VectorizedArrayType,
                                                            VectorizedArrayType>;

      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        *data);

      for (unsigned int cell = 0; cell < data->n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          const VectorizedArrayType *src_ptr   = src.data() + cell * n_q_points * n_components;
          VectorizedArrayType *      dst_ptr   = dst.data() + cell * n_q_points * n_components;
          VectorizedArrayType *      phi_grads = phi.begin_gradients();
          if (dim > 2)
            {
              for (unsigned int c = 0; c < n_components; ++c)
                Eval::template apply<2, true, false, 1>(
                  phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                  src_ptr + c * n_q_points,
                  phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points);
            }

          const std::array<Tensor<1, dim, VectorizedArrayType>,
                           GeometryInfo<dim>::vertices_per_cell> &v =
            cell_vertex_coefficients[cell];
          for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
            {
              using Eval2 =
                dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                         2,
                                                         n_q_points_1d,
                                                         n_q_points_1d,
                                                         VectorizedArrayType,
                                                         VectorizedArrayType>;
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval2::template apply<1, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    src_ptr + c * n_q_points + qz * n_q_points_2d,
                    phi_grads + (2 * c + 1) * n_q_points_2d);
                  Eval2::template apply<0, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    src_ptr + c * n_q_points + qz * n_q_points_2d,
                    phi_grads + 2 * c * n_q_points_2d);
                }

              for (unsigned int qy = 0; qy < n_q_points_1d; ++qy)
                {
                  const auto z = quad_1d.point(qz)[0];
                  const auto y = quad_1d.point(qy)[0];
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
                      const Number                        x = quad_1d.point(qx)[0];
                      Tensor<2, dim, VectorizedArrayType> jac;
                      jac[1] = y_con + x * y_var;
                      jac[2] = z_con + x * z_var;
                      for (unsigned int d = 0; d < dim; ++d)
                        jac[0][d] = x_con[d];
                      VectorizedArrayType det = do_invert(jac);
                      det                     = det * (q_weight_tmp * quad_1d.weight(qx));

                      for (unsigned int c = 0; c < n_components; ++c)
                        {
                          VectorizedArrayType tmp[dim], tmp2[dim];
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              tmp[d] =
                                jac[d][0] *
                                  phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] +
                                jac[d][1] *
                                  phi_grads[qy * n_q_points_1d + qx + (c * 2 + 1) * n_q_points_2d] +
                                jac[d][2] *
                                  phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points];
                              tmp[d] *= det;
                            }
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              tmp2[d] = jac[0][d] * tmp[0];
                              for (unsigned int e = 1; e < dim; ++e)
                                tmp2[d] += jac[e][d] * tmp[e];
                            }
                          phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] = tmp2[0];
                          phi_grads[qy * n_q_points_1d + qx + (c * 2 + 1) * n_q_points_2d] =
                            tmp2[1];
                          phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] =
                            tmp2[2];
                        }
                    }
                }

              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval2::template apply<0, false, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + 2 * c * n_q_points_2d,
                    dst_ptr + c * n_q_points + qz * n_q_points_2d);
                  Eval2::template apply<1, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + (2 * c + 1) * n_q_points_2d,
                    dst_ptr + c * n_q_points + qz * n_q_points_2d);
                }
            }
          for (unsigned int c = 0; c < n_components; ++c)
            {
              Eval::template apply<2, false, true, 1>(
                phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points,
                dst_ptr + c * n_q_points);
            }
        }
    }

    void
    apply_q2(AlignedVector<VectorizedArrayType> &      dst,
             const AlignedVector<VectorizedArrayType> &src) const
    {
      const std::pair<unsigned int, unsigned int> cell_range(0, data->n_cell_batches());
      EXPAND_OPERATIONS(apply_q2_impl);
    }

    template <int fe_degree, int n_q_points_1d>
    void
    apply_q2_impl(AlignedVector<VectorizedArrayType> &      dst,
                  const AlignedVector<VectorizedArrayType> &src,
                  const std::pair<unsigned int, unsigned int> &) const
    {
      AssertThrow(n_q_points_1d == fe_degree + 1, ExcNotImplemented());
      constexpr unsigned int n_q_points    = Utilities::pow(n_q_points_1d, dim);
      constexpr unsigned int n_q_points_2d = Utilities::pow(n_q_points_1d, 2);

      using TensorType = Tensor<1, dim, VectorizedArrayType>;
      using Eval = dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                            dim,
                                                            n_q_points_1d,
                                                            n_q_points_1d,
                                                            VectorizedArrayType,
                                                            VectorizedArrayType>;

      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        *data);

      using TensorType = Tensor<1, dim, VectorizedArrayType>;
      std::array<TensorType, Utilities::pow(3, dim - 1)> xi;
      std::array<TensorType, Utilities::pow(3, dim - 1)> di;

      for (unsigned int cell = 0; cell < data->n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          const VectorizedArrayType *src_ptr   = src.data() + cell * n_q_points * n_components;
          VectorizedArrayType *      dst_ptr   = dst.data() + cell * n_q_points * n_components;
          VectorizedArrayType *      phi_grads = phi.begin_gradients();
          if (dim > 2)
            {
              for (unsigned int c = 0; c < n_components; ++c)
                Eval::template apply<2, true, false, 1>(
                  phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                  src_ptr + c * n_q_points,
                  phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points);
            }

          const auto &v = cell_quadratic_coefficients[cell];

          for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
            {
              using Eval2 =
                dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                         2,
                                                         n_q_points_1d,
                                                         n_q_points_1d,
                                                         VectorizedArrayType,
                                                         VectorizedArrayType>;
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval2::template apply<1, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    src_ptr + c * n_q_points + qz * n_q_points_2d,
                    phi_grads + (2 * c + 1) * n_q_points_2d);
                  Eval2::template apply<0, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    src_ptr + c * n_q_points + qz * n_q_points_2d,
                    phi_grads + 2 * c * n_q_points_2d);
                }

              const Number z = quad_1d.point(qz)[0];
              di[0]          = v[9] + (z + z) * v[18];
              for (unsigned int i = 1; i < 9; ++i)
                {
                  xi[i] = v[i] + z * (v[9 + i] + z * v[18 + i]);
                  di[i] = v[9 + i] + (z + z) * v[18 + i];
                }
              for (unsigned int qy = 0; qy < n_q_points_1d; ++qy)
                {
                  const auto       y            = quad_1d.point(qy)[0];
                  const TensorType x1           = xi[1] + y * (xi[4] + y * xi[7]);
                  const TensorType x2           = xi[2] + y * (xi[5] + y * xi[8]);
                  const TensorType dy0          = xi[3] + (y + y) * xi[6];
                  const TensorType dy1          = xi[4] + (y + y) * xi[7];
                  const TensorType dy2          = xi[5] + (y + y) * xi[8];
                  const TensorType dz0          = di[0] + y * (di[3] + y * di[6]);
                  const TensorType dz1          = di[1] + y * (di[4] + y * di[7]);
                  const TensorType dz2          = di[2] + y * (di[5] + y * di[8]);
                  double           q_weight_tmp = quad_1d.weight(qz) * quad_1d.weight(qy);
                  for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                    {
                      const Number                        x = quad_1d.point(qx)[0];
                      Tensor<2, dim, VectorizedArrayType> jac;
                      jac[0]                  = x1 + (x + x) * x2;
                      jac[1]                  = dy0 + x * (dy1 + x * dy2);
                      jac[2]                  = dz0 + x * (dz1 + x * dz2);
                      VectorizedArrayType det = do_invert(jac);
                      det                     = det * (q_weight_tmp * quad_1d.weight(qx));

                      for (unsigned int c = 0; c < n_components; ++c)
                        {
                          VectorizedArrayType tmp[dim], tmp2[dim];
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              tmp[d] =
                                jac[d][0] *
                                  phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] +
                                jac[d][1] *
                                  phi_grads[qy * n_q_points_1d + qx + (c * 2 + 1) * n_q_points_2d] +
                                jac[d][2] *
                                  phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points];
                              tmp[d] *= det;
                            }
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              tmp2[d] = jac[0][d] * tmp[0];
                              for (unsigned int e = 1; e < dim; ++e)
                                tmp2[d] += jac[e][d] * tmp[e];
                            }
                          phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] = tmp2[0];
                          phi_grads[qy * n_q_points_1d + qx + (c * 2 + 1) * n_q_points_2d] =
                            tmp2[1];
                          phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] =
                            tmp2[2];
                        }
                    }
                }
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval2::template apply<0, false, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + 2 * c * n_q_points_2d,
                    dst_ptr + c * n_q_points + qz * n_q_points_2d);
                  Eval2::template apply<1, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + (2 * c + 1) * n_q_points_2d,
                    dst_ptr + c * n_q_points + qz * n_q_points_2d);
                }
            }
          for (unsigned int c = 0; c < n_components; ++c)
            {
              Eval::template apply<2, false, true, 1>(
                phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points,
                dst_ptr + c * n_q_points);
            }
        }
    }

    /**
     * Compute the diagonal (scalar variant) of the matrix
     */
    LinearAlgebra::distributed::Vector<Number>
    compute_inverse_diagonal() const
    {
      LinearAlgebra::distributed::Vector<Number> diag;
      data->initialize_dof_vector(diag);
      FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi(*data);

      AlignedVector<VectorizedArrayType> diagonal(phi.dofs_per_cell);
      for (unsigned int cell = 0; cell < data->n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                phi.submit_dof_value(VectorizedArrayType(), j);
              phi.submit_dof_value(make_vectorized_array<VectorizedArrayType>(1.), i);

              phi.evaluate(EvaluationFlags::gradients);
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                phi.submit_gradient(phi.get_gradient(q), q);
              phi.integrate(EvaluationFlags::gradients);
              diagonal[i] = phi.get_dof_value(i);
            }
          for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            phi.submit_dof_value(diagonal[i], i);
          phi.distribute_local_to_global(diag);
        }
      diag.compress(VectorOperation::add);
      for (unsigned int i = 0; i < diag.locally_owned_size(); ++i)
        if (diag.local_element(i) == 0.)
          diag.local_element(i) = 1.;
        else
          diag.local_element(i) = 1. / diag.local_element(i);
      return diag;
    }

  private:
    void
    local_apply_basic(const MatrixFree<dim, value_type, VectorizedArrayType> &,
                      VectorType &                                 dst,
                      const VectorType &                           src,
                      const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      EXPAND_OPERATIONS(local_apply_basic_impl);
    }

    template <int fe_degree, int n_q_points_1d>
    void
    local_apply_basic_impl(VectorType &                                 dst,
                           const VectorType &                           src,
                           const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      const auto &data = *this->data;
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        data);
      constexpr unsigned int n_read_components =
        IsBlockVector<VectorType>::value ? 1 : n_components;
      constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);

      using TensorType = Tensor<1, dim, VectorizedArrayType>;
      using Eval = dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                            dim,
                                                            n_q_points_1d,
                                                            n_q_points_1d,
                                                            VectorizedArrayType,
                                                            VectorizedArrayType>;

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          if (fe_degree > 2 && fast_read)
            for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
              read_dof_values_compressed<dim,
                                         fe_degree,
                                         n_q_points_1d,
                                         n_read_components,
                                         value_type>(
                ::internal::get_block(src, bl),
                compressed_dof_indices,
                all_indices_uniform,
                cell,
                phi.get_shape_info().data[0].shape_values_eo,
                phi.get_shape_info().data[0].element_type ==
                  dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation,
                phi.begin_values() + bl * n_q_points);
          else
            {
              phi.read_dof_values(src);
              phi.evaluate(EvaluationFlags::values);
            }
          VectorizedArrayType *phi_grads = phi.begin_gradients();
          if (dim == 2)
            {
              AssertThrow(false, ExcNotImplemented());
            }
          else if (dim == 3)
            {
              constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  if (fe_degree > 2 &&
                      phi.get_shape_info().data[0].element_type !=
                        dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation)
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             dim,
                                                             fe_degree + 1,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>::
                      template apply<2, true, false, 0>(
                        phi.get_shape_info().data[0].shape_values_eo.begin(),
                        phi.begin_values() + c * n_q_points,
                        phi.begin_values() + c * n_q_points);
                  Eval::template apply<2, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi.begin_values() + c * n_q_points,
                    phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points);
                }
              for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
                {
                  using Eval2 =
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             2,
                                                             n_q_points_1d,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>;
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      Eval2::template apply<1, true, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d,
                        phi_grads + (2 * c + 1) * n_q_points_2d);
                      Eval2::template apply<0, true, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d,
                        phi_grads + 2 * c * n_q_points_2d);
                    }
                  for (unsigned int qy = 0; qy < n_q_points_1d; ++qy)
                    {
                      double q_weight_tmp = quad_1d.weight(qz) * quad_1d.weight(qy);
                      if (static_cast<int>(data.get_mapping_info().cell_type[cell]) == 0)
                        {
                          const unsigned int ind =
                            data.get_mapping_info().cell_data[0].data_index_offsets[cell];
                          const VectorizedArrayType J =
                            data.get_mapping_info().cell_data[0].JxW_values[ind];
                          const Tensor<2, dim, VectorizedArrayType> jac =
                            data.get_mapping_info().cell_data[0].jacobians[0][ind];
                          for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                            {
                              const VectorizedArrayType det =
                                J * (q_weight_tmp * quad_1d.weight(qx));

                              for (unsigned int c = 0; c < n_components; ++c)
                                {
                                  VectorizedArrayType tmp[dim], tmp2[dim];
                                  for (unsigned int d = 0; d < dim; ++d)
                                    {
                                      tmp[d] =
                                        jac[d][0] * phi_grads[qy * n_q_points_1d + qx +
                                                              c * 2 * n_q_points_2d] +
                                        jac[d][1] * phi_grads[qy * n_q_points_1d + qx +
                                                              (c * 2 + 1) * n_q_points_2d] +
                                        jac[d][2] * phi_grads[q + 2 * n_components * n_q_points_2d +
                                                              c * n_q_points];
                                      tmp[d] *= det;
                                    }
                                  for (unsigned int d = 0; d < dim; ++d)
                                    {
                                      tmp2[d] = jac[0][d] * tmp[0];
                                      for (unsigned int e = 1; e < dim; ++e)
                                        tmp2[d] += jac[e][d] * tmp[e];
                                    }
                                  phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] =
                                    tmp2[0];
                                  phi_grads[qy * n_q_points_1d + qx + (c * 2 + 1) * n_q_points_2d] =
                                    tmp2[1];
                                  phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] =
                                    tmp2[2];
                                }
                            }
                        }
                      else
                        {
                          for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                            {
                              const VectorizedArrayType det = phi.JxW(q);
                              const auto                jac = phi.inverse_jacobian(q);

                              for (unsigned int c = 0; c < n_components; ++c)
                                {
                                  VectorizedArrayType tmp[dim], tmp2[dim];
                                  for (unsigned int d = 0; d < dim; ++d)
                                    {
                                      tmp[d] =
                                        jac[d][0] * phi_grads[qy * n_q_points_1d + qx +
                                                              c * 2 * n_q_points_2d] +
                                        jac[d][1] * phi_grads[qy * n_q_points_1d + qx +
                                                              (c * 2 + 1) * n_q_points_2d] +
                                        jac[d][2] * phi_grads[q + 2 * n_components * n_q_points_2d +
                                                              c * n_q_points];
                                      tmp[d] *= det;
                                    }
                                  for (unsigned int d = 0; d < dim; ++d)
                                    {
                                      tmp2[d] = jac[0][d] * tmp[0];
                                      for (unsigned int e = 1; e < dim; ++e)
                                        tmp2[d] += jac[e][d] * tmp[e];
                                    }
                                  phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] =
                                    tmp2[0];
                                  phi_grads[qy * n_q_points_1d + qx + (c * 2 + 1) * n_q_points_2d] =
                                    tmp2[1];
                                  phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] =
                                    tmp2[2];
                                }
                            }
                        }
                    }
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      Eval2::template apply<0, false, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi_grads + 2 * c * n_q_points_2d,
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d);
                      Eval2::template apply<1, false, true, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi_grads + (2 * c + 1) * n_q_points_2d,
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d);
                    }
                }
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<2, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points,
                    phi.begin_values() + c * n_q_points);
                  if (fe_degree > 2 && fast_read &&
                      phi.get_shape_info().data[0].element_type !=
                        dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation)
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             dim,
                                                             fe_degree + 1,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>::
                      template apply<2, false, false, 0>(
                        phi.get_shape_info().data[0].shape_values_eo.begin(),
                        phi.begin_values() + c * n_q_points,
                        phi.begin_values() + c * n_q_points);
                }
            }

          if (fe_degree > 2 && fast_read)
            for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
              distribute_local_to_global_compressed<dim,
                                                    fe_degree,
                                                    n_q_points_1d,
                                                    n_read_components,
                                                    Number>(
                ::internal::get_block(dst, bl),
                compressed_dof_indices,
                all_indices_uniform,
                cell,
                phi.get_shape_info().data[0].shape_values_eo,
                phi.get_shape_info().data[0].element_type ==
                  dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation,
                phi.begin_values() + bl * n_q_points);
          else
            phi.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    template <bool do_sum>
    void
    local_apply_linear_geo(const MatrixFree<dim, value_type, VectorizedArrayType> &,
                           VectorType &                                 dst,
                           const VectorType &                           src,
                           const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      static_assert(do_sum == false, "Embedded summation not implemented");
      EXPAND_OPERATIONS(local_apply_linear_geo_impl);
    }

    template <int fe_degree, int n_q_points_1d>
    void
    local_apply_linear_geo_impl(VectorType &                                 dst,
                                const VectorType &                           src,
                                const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      constexpr bool         do_sum = false;
      constexpr unsigned int n_read_components =
        IsBlockVector<VectorType>::value ? 1 : n_components;
      const auto &data = *this->data;
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        data);
      constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);
      using Eval = dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                            dim,
                                                            n_q_points_1d,
                                                            n_q_points_1d,
                                                            VectorizedArrayType,
                                                            VectorizedArrayType>;

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          if (fe_degree > 2 && fast_read)
            for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
              read_dof_values_compressed<dim,
                                         fe_degree,
                                         n_q_points_1d,
                                         n_read_components,
                                         value_type>(
                ::internal::get_block(src, bl),
                compressed_dof_indices,
                all_indices_uniform,
                cell,
                phi.get_shape_info().data[0].shape_values_eo,
                phi.get_shape_info().data[0].element_type ==
                  dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation,
                phi.begin_values() + bl * n_q_points);
          else
            {
              phi.read_dof_values(src);
              phi.evaluate(EvaluationFlags::values);
            }
          const std::array<Tensor<1, dim, VectorizedArrayType>,
                           GeometryInfo<dim>::vertices_per_cell> &v =
            cell_vertex_coefficients[cell];
          VectorizedArrayType *phi_grads = phi.begin_gradients();
          if (dim == 2)
            {
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<1, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi.begin_values() + c * n_q_points,
                    phi_grads + n_q_points + 2 * c * n_q_points);
                  Eval::template apply<0, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi.begin_values(),
                    phi_grads + 2 * c * n_q_points);
                }
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
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<0, false, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + 2 * c * n_q_points,
                    phi.begin_values());
                  Eval::template apply<1, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + n_q_points + 2 * c * n_q_points,
                    phi.begin_values() + c * n_q_points);
                }
            }
          else if (dim == 3)
            {
              constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  if (fe_degree > 2 &&
                      phi.get_shape_info().data[0].element_type !=
                        dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation)
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             dim,
                                                             fe_degree + 1,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>::
                      template apply<2, true, false, 0>(
                        phi.get_shape_info().data[0].shape_values_eo.begin(),
                        phi.begin_values() + c * n_q_points,
                        phi.begin_values() + c * n_q_points);
                  Eval::template apply<2, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi.begin_values() + c * n_q_points,
                    phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points);
                }
              for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
                {
                  using Eval2 =
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             2,
                                                             n_q_points_1d,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>;
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      Eval2::template apply<1, true, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d,
                        phi_grads + (2 * c + 1) * n_q_points_2d);
                      Eval2::template apply<0, true, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d,
                        phi_grads + 2 * c * n_q_points_2d);
                    }
                  for (unsigned int qy = 0; qy < n_q_points_1d; ++qy)
                    {
                      const auto z = quad_1d.point(qz)[0];
                      const auto y = quad_1d.point(qy)[0];
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
                          const Number                        x = quad_1d.point(qx)[0];
                          Tensor<2, dim, VectorizedArrayType> jac;
                          jac[1] = y_con + x * y_var;
                          jac[2] = z_con + x * z_var;
                          for (unsigned int d = 0; d < dim; ++d)
                            jac[0][d] = x_con[d];
                          VectorizedArrayType det = do_invert(jac);
                          det                     = det * (q_weight_tmp * quad_1d.weight(qx));

                          for (unsigned int c = 0; c < n_components; ++c)
                            {
                              VectorizedArrayType tmp[dim], tmp2[dim];
                              for (unsigned int d = 0; d < dim; ++d)
                                {
                                  tmp[d] =
                                    jac[d][0] *
                                      phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] +
                                    jac[d][1] * phi_grads[qy * n_q_points_1d + qx +
                                                          (c * 2 + 1) * n_q_points_2d] +
                                    jac[d][2] * phi_grads[q + 2 * n_components * n_q_points_2d +
                                                          c * n_q_points];
                                  tmp[d] *= det;
                                }
                              for (unsigned int d = 0; d < dim; ++d)
                                {
                                  tmp2[d] = jac[0][d] * tmp[0];
                                  for (unsigned int e = 1; e < dim; ++e)
                                    tmp2[d] += jac[e][d] * tmp[e];
                                }
                              phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] = tmp2[0];
                              phi_grads[qy * n_q_points_1d + qx + (c * 2 + 1) * n_q_points_2d] =
                                tmp2[1];
                              phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] =
                                tmp2[2];
                            }
                        }
                    }
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      Eval2::template apply<0, false, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi_grads + 2 * c * n_q_points_2d,
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d);
                      Eval2::template apply<1, false, true, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi_grads + (2 * c + 1) * n_q_points_2d,
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d);
                    }
                }
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<2, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points,
                    phi.begin_values() + c * n_q_points);
                  if (!do_sum && fe_degree > 2 && fast_read &&
                      phi.get_shape_info().data[0].element_type !=
                        dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation)
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             dim,
                                                             fe_degree + 1,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>::
                      template apply<2, false, false, 0>(
                        phi.get_shape_info().data[0].shape_values_eo.begin(),
                        phi.begin_values() + c * n_q_points,
                        phi.begin_values() + c * n_q_points);
                }
            }
          VectorizedArrayType
            scratch[do_sum ? Utilities::pow(fe_degree + 1, dim) * n_components : 1];
          if (do_sum)
            {
              phi.read_dof_values(src);
              for (unsigned int i = 0; i < Utilities::pow(fe_degree + 1, dim) * n_components; ++i)
                scratch[i] = phi.begin_dof_values()[i];
            }

          if (!do_sum && fe_degree > 2 && fast_read)
            for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
              distribute_local_to_global_compressed<dim,
                                                    fe_degree,
                                                    n_q_points_1d,
                                                    n_read_components,
                                                    Number>(
                ::internal::get_block(dst, bl),
                compressed_dof_indices,
                all_indices_uniform,
                cell,
                phi.get_shape_info().data[0].shape_values_eo,
                phi.get_shape_info().data[0].element_type ==
                  dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation,
                phi.begin_values() + bl * n_q_points);
          else
            {
              phi.integrate_scatter(EvaluationFlags::values, dst);
              if (do_sum)
                {
                  VectorizedArrayType local_sum = VectorizedArrayType();
                  for (unsigned int i = 0; i < Utilities::pow(fe_degree + 1, dim) * n_components;
                       ++i)
                    local_sum += phi.begin_dof_values()[i] * scratch[i];

                  for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
                    accumulated_sum += local_sum[v];
                }
            }
        }
    }

    void
    local_apply_quadratic_geo(const MatrixFree<dim, value_type, VectorizedArrayType> &,
                              VectorType &                                 dst,
                              const VectorType &                           src,
                              const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      EXPAND_OPERATIONS(local_apply_quadratic_geo_impl);
    }

    template <int fe_degree, int n_q_points_1d>
    void
    local_apply_quadratic_geo_impl(VectorType &                                 dst,
                                   const VectorType &                           src,
                                   const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      const auto &data = *this->data;
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        data);
      constexpr unsigned int n_read_components =
        IsBlockVector<VectorType>::value ? 1 : n_components;
      constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);

      using TensorType = Tensor<1, dim, VectorizedArrayType>;
      std::array<TensorType, Utilities::pow(3, dim - 1)> xi;
      std::array<TensorType, Utilities::pow(3, dim - 1)> di;
      using Eval = dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                            dim,
                                                            n_q_points_1d,
                                                            n_q_points_1d,
                                                            VectorizedArrayType,
                                                            VectorizedArrayType>;

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          if (fe_degree > 2 && fast_read)
            for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
              read_dof_values_compressed<dim,
                                         fe_degree,
                                         n_q_points_1d,
                                         n_read_components,
                                         value_type>(
                ::internal::get_block(src, bl),
                compressed_dof_indices,
                all_indices_uniform,
                cell,
                phi.get_shape_info().data[0].shape_values_eo,
                phi.get_shape_info().data[0].element_type ==
                  dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation,
                phi.begin_values() + bl * n_q_points);
          else
            {
              phi.read_dof_values(src);
              phi.evaluate(EvaluationFlags::values);
            }
          const auto &         v         = cell_quadratic_coefficients[cell];
          VectorizedArrayType *phi_grads = phi.begin_gradients();
          if (dim == 2)
            {
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<1, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi.begin_values() + c * n_q_points,
                    phi_grads + n_q_points + 2 * c * n_q_points);
                  Eval::template apply<0, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi.begin_values(),
                    phi_grads + 2 * c * n_q_points);
                }
              for (unsigned int q = 0, qy = 0; qy < n_q_points_1d; ++qy)
                {
                  const Number     y  = quad_1d.point(qy)[0];
                  const TensorType x1 = v[1] + y * (v[4] + y * v[7]);
                  const TensorType x2 = v[2] + y * (v[5] + y * v[8]);
                  const TensorType d0 = v[3] + (y + y) * v[6];
                  const TensorType d1 = v[4] + (y + y) * v[7];
                  const TensorType d2 = v[5] + (y + y) * v[8];
                  for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                    {
                      const Number q_weight = quad_1d.weight(qy) * quad_1d.weight(qx);
                      const Number x        = quad_1d.point(qx)[0];
                      Tensor<2, dim, VectorizedArrayType> jac;
                      jac[0]                        = x1 + (x + x) * x2;
                      jac[1]                        = d0 + x * d1 + (x * x) * d2;
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
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<0, false, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + 2 * c * n_q_points,
                    phi.begin_values());
                  Eval::template apply<1, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + n_q_points + 2 * c * n_q_points,
                    phi.begin_values() + c * n_q_points);
                }
            }
          else if (dim == 3)
            {
              constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  if (fe_degree > 2 &&
                      phi.get_shape_info().data[0].element_type !=
                        dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation)
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             dim,
                                                             fe_degree + 1,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>::
                      template apply<2, true, false, 0>(
                        phi.get_shape_info().data[0].shape_values_eo.begin(),
                        phi.begin_values() + c * n_q_points,
                        phi.begin_values() + c * n_q_points);
                  Eval::template apply<2, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi.begin_values() + c * n_q_points,
                    phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points);
                }
              for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
                {
                  using Eval2 =
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             2,
                                                             n_q_points_1d,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>;
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      Eval2::template apply<1, true, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d,
                        phi_grads + (2 * c + 1) * n_q_points_2d);
                      Eval2::template apply<0, true, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d,
                        phi_grads + 2 * c * n_q_points_2d);
                    }
                  const Number z = quad_1d.point(qz)[0];
                  di[0]          = v[9] + (z + z) * v[18];
                  for (unsigned int i = 1; i < 9; ++i)
                    {
                      xi[i] = v[i] + z * (v[9 + i] + z * v[18 + i]);
                      di[i] = v[9 + i] + (z + z) * v[18 + i];
                    }
                  for (unsigned int qy = 0; qy < n_q_points_1d; ++qy)
                    {
                      const auto       y            = quad_1d.point(qy)[0];
                      const TensorType x1           = xi[1] + y * (xi[4] + y * xi[7]);
                      const TensorType x2           = xi[2] + y * (xi[5] + y * xi[8]);
                      const TensorType dy0          = xi[3] + (y + y) * xi[6];
                      const TensorType dy1          = xi[4] + (y + y) * xi[7];
                      const TensorType dy2          = xi[5] + (y + y) * xi[8];
                      const TensorType dz0          = di[0] + y * (di[3] + y * di[6]);
                      const TensorType dz1          = di[1] + y * (di[4] + y * di[7]);
                      const TensorType dz2          = di[2] + y * (di[5] + y * di[8]);
                      double           q_weight_tmp = quad_1d.weight(qz) * quad_1d.weight(qy);
                      for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                        {
                          const Number                        x = quad_1d.point(qx)[0];
                          Tensor<2, dim, VectorizedArrayType> jac;
                          jac[0]                  = x1 + (x + x) * x2;
                          jac[1]                  = dy0 + x * (dy1 + x * dy2);
                          jac[2]                  = dz0 + x * (dz1 + x * dz2);
                          VectorizedArrayType det = do_invert(jac);
                          det                     = det * (q_weight_tmp * quad_1d.weight(qx));

                          for (unsigned int c = 0; c < n_components; ++c)
                            {
                              VectorizedArrayType tmp[dim], tmp2[dim];
                              for (unsigned int d = 0; d < dim; ++d)
                                {
                                  tmp[d] =
                                    jac[d][0] *
                                      phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] +
                                    jac[d][1] * phi_grads[qy * n_q_points_1d + qx +
                                                          (c * 2 + 1) * n_q_points_2d] +
                                    jac[d][2] * phi_grads[q + 2 * n_components * n_q_points_2d +
                                                          c * n_q_points];
                                  tmp[d] *= det;
                                }
                              for (unsigned int d = 0; d < dim; ++d)
                                {
                                  tmp2[d] = jac[0][d] * tmp[0];
                                  for (unsigned int e = 1; e < dim; ++e)
                                    tmp2[d] += jac[e][d] * tmp[e];
                                }
                              phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] = tmp2[0];
                              phi_grads[qy * n_q_points_1d + qx + (c * 2 + 1) * n_q_points_2d] =
                                tmp2[1];
                              phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] =
                                tmp2[2];
                            }
                        }
                    }
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      Eval2::template apply<0, false, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi_grads + 2 * c * n_q_points_2d,
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d);
                      Eval2::template apply<1, false, true, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi_grads + (2 * c + 1) * n_q_points_2d,
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d);
                    }
                }
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<2, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points,
                    phi.begin_values() + c * n_q_points);
                  if (fe_degree > 2 && fast_read &&
                      phi.get_shape_info().data[0].element_type !=
                        dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation)
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             dim,
                                                             fe_degree + 1,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>::
                      template apply<2, false, false, 0>(
                        phi.get_shape_info().data[0].shape_values_eo.begin(),
                        phi.begin_values() + c * n_q_points,
                        phi.begin_values() + c * n_q_points);
                }
            }

          if (fe_degree > 2 && fast_read)
            for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
              distribute_local_to_global_compressed<dim,
                                                    fe_degree,
                                                    n_q_points_1d,
                                                    n_read_components,
                                                    Number>(
                ::internal::get_block(dst, bl),
                compressed_dof_indices,
                all_indices_uniform,
                cell,
                phi.get_shape_info().data[0].shape_values_eo,
                phi.get_shape_info().data[0].element_type ==
                  dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation,
                phi.begin_values() + bl * n_q_points);
          else
            phi.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    void
    local_apply_merged(const MatrixFree<dim, value_type, VectorizedArrayType> &,
                       VectorType &                                 dst,
                       const VectorType &                           src,
                       const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      EXPAND_OPERATIONS(local_apply_merged_impl);
    }

    template <int fe_degree, int n_q_points_1d>
    void
    local_apply_merged_impl(VectorType &                                 dst,
                            const VectorType &                           src,
                            const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      constexpr unsigned int n_read_components =
        IsBlockVector<VectorType>::value ? 1 : n_components;
      constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);

      const auto &data = *this->data;
      using TensorType = Tensor<1, dim, VectorizedArrayType>;
      using Eval = dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                            dim,
                                                            n_q_points_1d,
                                                            n_q_points_1d,
                                                            VectorizedArrayType,
                                                            VectorizedArrayType>;

      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        data);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          if (fe_degree > 2 && fast_read)
            for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
              read_dof_values_compressed<dim,
                                         fe_degree,
                                         n_q_points_1d,
                                         n_read_components,
                                         value_type>(
                ::internal::get_block(src, bl),
                compressed_dof_indices,
                all_indices_uniform,
                cell,
                phi.get_shape_info().data[0].shape_values_eo,
                phi.get_shape_info().data[0].element_type ==
                  dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation,
                phi.begin_values() + bl * n_q_points);
          else
            {
              phi.read_dof_values(src);
              phi.evaluate(EvaluationFlags::values);
            }
          VectorizedArrayType *phi_grads = phi.begin_gradients();
          if (dim == 2)
            {
              AssertThrow(false, ExcNotImplemented());
            }
          else if (dim == 3)
            {
              constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  if (fe_degree > 2 &&
                      phi.get_shape_info().data[0].element_type !=
                        dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation)
                    {
                      dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                               dim,
                                                               fe_degree + 1,
                                                               n_q_points_1d,
                                                               VectorizedArrayType,
                                                               VectorizedArrayType>::
                        template apply<2, true, false, 0>(
                          phi.get_shape_info().data[0].shape_values_eo.begin(),
                          phi.begin_values() + c * n_q_points,
                          phi.begin_values() + c * n_q_points);
                    }

                  Eval::template apply<2, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi.begin_values() + c * n_q_points,
                    phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points);
                }
              for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
                {
                  using Eval2 =
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             2,
                                                             n_q_points_1d,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>;
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      Eval2::template apply<1, true, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d,
                        phi_grads + (2 * c + 1) * n_q_points_2d);
                      Eval2::template apply<0, true, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d,
                        phi_grads + 2 * c * n_q_points_2d);
                    }
                  for (unsigned int qxy = 0; qxy < n_q_points_2d; ++qxy, ++q)
                    for (unsigned int c = 0; c < n_components; ++c)
                      {
                        VectorizedArrayType tmp0 = phi_grads[qxy + c * 2 * n_q_points_2d];
                        VectorizedArrayType tmp1 = phi_grads[qxy + (c * 2 + 1) * n_q_points_2d];
                        phi_grads[qxy + c * 2 * n_q_points_2d] =
                          (merged_coefficients[cell * n_q_points + q][0] * tmp0 +
                           merged_coefficients[cell * n_q_points + q][1] * tmp1 +
                           merged_coefficients[cell * n_q_points + q][2] *
                             phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points]);
                        phi_grads[qxy + (c * 2 + 1) * n_q_points_2d] =
                          (merged_coefficients[cell * n_q_points + q][1] * tmp0 +
                           merged_coefficients[cell * n_q_points + q][3] * tmp1 +
                           merged_coefficients[cell * n_q_points + q][4] *
                             phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points]);
                        phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] =
                          (merged_coefficients[cell * n_q_points + q][2] * tmp0 +
                           merged_coefficients[cell * n_q_points + q][4] * tmp1 +
                           merged_coefficients[cell * n_q_points + q][5] *
                             phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points]);
                      }
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      Eval2::template apply<0, false, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi_grads + 2 * c * n_q_points_2d,
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d);
                      Eval2::template apply<1, false, true, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi_grads + (2 * c + 1) * n_q_points_2d,
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d);
                    }
                }
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<2, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points,
                    phi.begin_values() + c * n_q_points);
                  if (fe_degree > 2 && fast_read &&
                      phi.get_shape_info().data[0].element_type !=
                        dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation)
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             dim,
                                                             fe_degree + 1,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>::
                      template apply<2, false, false, 0>(
                        phi.get_shape_info().data[0].shape_values_eo.begin(),
                        phi.begin_values() + c * n_q_points,
                        phi.begin_values() + c * n_q_points);
                }
            }

          if (fe_degree > 2 && fast_read)
            for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
              distribute_local_to_global_compressed<dim,
                                                    fe_degree,
                                                    n_q_points_1d,
                                                    n_read_components,
                                                    Number>(
                ::internal::get_block(dst, bl),
                compressed_dof_indices,
                all_indices_uniform,
                cell,
                phi.get_shape_info().data[0].shape_values_eo,
                phi.get_shape_info().data[0].element_type ==
                  dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation,
                phi.begin_values() + bl * n_q_points);
          else
            phi.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    void
    local_apply_construct_q(const MatrixFree<dim, value_type, VectorizedArrayType> &,
                            VectorType &                                 dst,
                            const VectorType &                           src,
                            const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      EXPAND_OPERATIONS(local_apply_construct_q_impl);
    }

    template <int fe_degree, int n_q_points_1d>
    void
    local_apply_construct_q_impl(VectorType &                                 dst,
                                 const VectorType &                           src,
                                 const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      const auto &data = *this->data;
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        data);
      // constexpr unsigned int n_read_components =
      //  IsBlockVector<VectorType>::value ? 1 : n_components;
      constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);
      VectorizedArrayType    jacobians_z[dim * n_q_points];
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          // if (fe_degree > 2)
          //  for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
          //    read_dof_values_compressed<dim, fe_degree, n_read_components, value_type>(
          //      ::internal::get_block(src, bl),
          //      compressed_dof_indices,
          //      all_indices_uniform,
          //      cell,
          //      phi.begin_dof_values() + bl * Utilities::pow(fe_degree + 1, dim));
          // else
          phi.read_dof_values(src);
          phi.evaluate(EvaluationFlags::gradients);
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
          phi.integrate(EvaluationFlags::gradients);
          // if (fe_degree > 2)
          //  for (unsigned int bl = 0; bl < ::internal::get_n_blocks(src); ++bl)
          //    distribute_local_to_global_compressed<dim, fe_degree, n_read_components, Number>(
          //      ::internal::get_block(dst, bl),
          //      compressed_dof_indices,
          //      all_indices_uniform,
          //      cell,
          //      phi.begin_dof_values() + bl * Utilities::pow(fe_degree + 1, dim));
          // else
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

    // For local_apply_quadratic_geo
    AlignedVector<std::array<Tensor<1, dim, VectorizedArrayType>, Utilities::pow(3, dim)>>
      cell_quadratic_coefficients;

    // For local_apply_merged: dim*(dim+1)/2 coefficients
    AlignedVector<Tensor<1, (dim * (dim + 1) / 2), VectorizedArrayType>> merged_coefficients;

    // for local_apply_construct_q:
    // Data in all quadrature points in collocated space
    AlignedVector<VectorizedArrayType> quadrature_points;

    AlignedVector<Number> coefficients_eo;

    std::vector<unsigned int>  compressed_dof_indices;
    std::vector<unsigned char> all_indices_uniform;

    bool fast_read;
  };

#undef EXPAND_OPERATIONS

} // namespace Poisson

#endif
