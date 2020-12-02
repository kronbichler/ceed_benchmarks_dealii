
#ifndef vector_access_reduced_h
#define vector_access_reduced_h

#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#ifdef USE_STD_SIMD

namespace dealii
{
  template <typename Number, typename X>
  inline DEAL_II_ALWAYS_INLINE void
  vectorized_load_and_transpose(const unsigned int                                  n_entries,
                                const Number *                                      in,
                                const unsigned int *                                offsets,
                                std::experimental::parallelism_v2::simd<Number, X> *out)
  {
    for (unsigned int i = 0; i < n_entries; ++i)
      for (unsigned int v = 0; v < std::experimental::parallelism_v2::simd<Number, X>::size(); ++v)
        out[i][v] = in[offsets[v] + i];
  }

  template <typename Number, typename X>
  inline DEAL_II_ALWAYS_INLINE void
  vectorized_transpose_and_store(const bool         add_into,
                                 const unsigned int n_entries,
                                 const std::experimental::parallelism_v2::simd<Number, X> *in,
                                 const unsigned int *                                      offsets,
                                 Number *                                                  out)
  {
    if (add_into)
      for (unsigned int i = 0; i < n_entries; ++i)
        for (unsigned int v = 0; v < std::experimental::parallelism_v2::simd<Number, X>::size();
             ++v)
          out[offsets[v] + i] += in[i][v];
    else
      for (unsigned int i = 0; i < n_entries; ++i)
        for (unsigned int v = 0; v < std::experimental::parallelism_v2::simd<Number, X>::size();
             ++v)
          out[offsets[v] + i] = in[i][v];
  }

} // namespace dealii

#endif

template <int dim, int fe_degree, int n_components, typename Number, typename VectorizedArrayType>
void
read_dof_values_compressed(const dealii::LinearAlgebra::distributed::Vector<Number> &vec,
                           const std::vector<unsigned int> & compressed_indices,
                           const std::vector<unsigned char> &all_indices_unconstrained,
                           const unsigned int                cell_no,
                           VectorizedArrayType *             dof_values)
{
  AssertIndexRange(cell_no * dealii::Utilities::pow(3, dim) * VectorizedArrayType::size(),
                   compressed_indices.size());
  constexpr unsigned int n_lanes = VectorizedArrayType::size();
  const unsigned int *   cell_indices =
    compressed_indices.data() + cell_no * n_lanes * dealii::Utilities::pow(3, dim);
  const unsigned char *cell_unconstrained =
    all_indices_unconstrained.data() + cell_no * dealii::Utilities::pow(3, dim);
  constexpr unsigned int dofs_per_comp = dealii::Utilities::pow(fe_degree + 1, dim);
  dealii::internal::VectorReader<Number, VectorizedArrayType> reader;
  for (unsigned int i2 = 0, i = 0, compressed_i2 = 0, offset_i2 = 0;
       i2 < (dim == 3 ? (fe_degree + 1) : 1);
       ++i2)
    {
      for (unsigned int i1 = 0, compressed_i1 = 0, offset_i1 = 0;
           i1 < (dim > 1 ? (fe_degree + 1) : 1);
           ++i1)
        {
          const unsigned int offset =
            (compressed_i1 == 1 ? fe_degree - 1 : 1) * offset_i2 + offset_i1;
          const unsigned int *indices =
            cell_indices + 3 * n_lanes * (compressed_i2 * 3 + compressed_i1);
          const unsigned char *unconstrained =
            cell_unconstrained + 3 * (compressed_i2 * 3 + compressed_i1);

          // left end point
          if (unconstrained[0])
            for (unsigned int c = 0; c < n_components; ++c)
              reader.process_dof_gather(indices,
                                        vec,
                                        offset * n_components + c,
                                        dof_values[i + c * dofs_per_comp],
                                        std::integral_constant<bool, true>());
          else
            for (unsigned int v = 0; v < n_lanes; ++v)
              for (unsigned int c = 0; c < n_components; ++c)
                dof_values[i + c * dofs_per_comp][v] =
                  (indices[v] == dealii::numbers::invalid_unsigned_int) ?
                    0. :
                    vec.local_element(indices[v] + offset * n_components + c);
          ++i;
          indices += n_lanes;

          // interior points of line
          if (unconstrained[1])
            {
              VectorizedArrayType    tmp[fe_degree > 1 ? (fe_degree - 1) * n_components : 1];
              constexpr unsigned int n_regular = (fe_degree - 1) * n_components / 4 * 4;
              dealii::vectorized_load_and_transpose(
                n_regular, vec.begin() + offset * (fe_degree - 1) * n_components, indices, tmp);
              for (unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
                reader.process_dof_gather(indices,
                                          vec,
                                          offset * (fe_degree - 1) * n_components + i0,
                                          tmp[i0],
                                          std::integral_constant<bool, true>());
              for (unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                for (unsigned int c = 0; c < n_components; ++c)
                  dof_values[i + c * dofs_per_comp] = tmp[i0 * n_components + c];
            }
          else
            for (unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
              for (unsigned int v = 0; v < n_lanes; ++v)
                if (indices[v] != dealii::numbers::invalid_unsigned_int)
                  for (unsigned int c = 0; c < n_components; ++c)
                    dof_values[i + c * dofs_per_comp][v] = vec.local_element(
                      indices[v] + (offset * (fe_degree - 1) + i0) * n_components + c);
                else
                  for (unsigned int c = 0; c < n_components; ++c)
                    dof_values[i + c * dofs_per_comp][v] = 0.;
          indices += n_lanes;

          // right end point
          if (unconstrained[2])
            for (unsigned int c = 0; c < n_components; ++c)
              reader.process_dof_gather(indices,
                                        vec,
                                        offset * n_components + c,
                                        dof_values[i + c * dofs_per_comp],
                                        std::integral_constant<bool, true>());
          else
            for (unsigned int v = 0; v < n_lanes; ++v)
              for (unsigned int c = 0; c < n_components; ++c)
                dof_values[i + c * dofs_per_comp][v] =
                  (indices[v] == dealii::numbers::invalid_unsigned_int) ?
                    0. :
                    vec.local_element(indices[v] + offset * n_components + c);
          ++i;

          if (i1 == 0 || i1 == fe_degree - 1)
            {
              ++compressed_i1;
              offset_i1 = 0;
            }
          else
            ++offset_i1;
        }
      if (i2 == 0 || i2 == fe_degree - 1)
        {
          ++compressed_i2;
          offset_i2 = 0;
        }
      else
        ++offset_i2;
    }
}



template <int dim, int fe_degree, int n_components, typename Number, typename VectorizedArrayType>
void
distribute_local_to_global_compressed(dealii::LinearAlgebra::distributed::Vector<Number> &vec,
                                      const std::vector<unsigned int> & compressed_indices,
                                      const std::vector<unsigned char> &all_indices_unconstrained,
                                      const unsigned int                cell_no,
                                      VectorizedArrayType *             dof_values)
{
  AssertIndexRange(cell_no * dealii::Utilities::pow(3, dim) * VectorizedArrayType::size(),
                   compressed_indices.size());
  constexpr unsigned int n_lanes = VectorizedArrayType::size();
  const unsigned int *   cell_indices =
    compressed_indices.data() + cell_no * n_lanes * dealii::Utilities::pow(3, dim);
  const unsigned char *cell_unconstrained =
    all_indices_unconstrained.data() + cell_no * dealii::Utilities::pow(3, dim);
  constexpr unsigned int dofs_per_comp = dealii::Utilities::pow(fe_degree + 1, dim);
  dealii::internal::VectorDistributorLocalToGlobal<Number, VectorizedArrayType> distributor;

  for (unsigned int i2 = 0, i = 0, compressed_i2 = 0, offset_i2 = 0;
       i2 < (dim == 3 ? (fe_degree + 1) : 1);
       ++i2)
    {
      for (unsigned int i1 = 0, compressed_i1 = 0, offset_i1 = 0;
           i1 < (dim > 1 ? (fe_degree + 1) : 1);
           ++i1)
        {
          const unsigned int offset =
            (compressed_i1 == 1 ? fe_degree - 1 : 1) * offset_i2 + offset_i1;
          const unsigned int *indices =
            cell_indices + 3 * n_lanes * (compressed_i2 * 3 + compressed_i1);
          const unsigned char *unconstrained =
            cell_unconstrained + 3 * (compressed_i2 * 3 + compressed_i1);

          // left end point
          if (unconstrained[0])
            for (unsigned int c = 0; c < n_components; ++c)
              distributor.process_dof_gather(indices,
                                             vec,
                                             offset * n_components + c,
                                             dof_values[i + c * dofs_per_comp],
                                             std::integral_constant<bool, true>());
          else
            for (unsigned int v = 0; v < n_lanes; ++v)
              for (unsigned int c = 0; c < n_components; ++c)
                if (indices[v] != dealii::numbers::invalid_unsigned_int)
                  vec.local_element(indices[v] + offset * n_components + c) +=
                    dof_values[i + c * dofs_per_comp][v];
          ++i;
          indices += n_lanes;

          // interior points of line
          if (unconstrained[1])
            {
              VectorizedArrayType tmp[fe_degree > 1 ? (fe_degree - 1) * n_components : 1];
              for (unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                for (unsigned int c = 0; c < n_components; ++c)
                  tmp[i0 * n_components + c] = dof_values[i + c * dofs_per_comp];

              constexpr unsigned int n_regular = (fe_degree - 1) * n_components / 4 * 4;
              dealii::vectorized_transpose_and_store(true,
                                                     n_regular,
                                                     tmp,
                                                     indices,
                                                     vec.begin() +
                                                       offset * (fe_degree - 1) * n_components);
              for (unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
                distributor.process_dof_gather(indices,
                                               vec,
                                               offset * (fe_degree - 1) * n_components + i0,
                                               tmp[i0],
                                               std::integral_constant<bool, true>());
            }
          else
            for (unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
              for (unsigned int v = 0; v < n_lanes; ++v)
                if (indices[v] != dealii::numbers::invalid_unsigned_int)
                  for (unsigned int c = 0; c < n_components; ++c)
                    vec.local_element(indices[v] + (offset * (fe_degree - 1) + i0) * n_components +
                                      c) += dof_values[i + c * dofs_per_comp][v];
          indices += n_lanes;

          // right end point
          if (unconstrained[2])
            for (unsigned int c = 0; c < n_components; ++c)
              distributor.process_dof_gather(indices,
                                             vec,
                                             offset * n_components + c,
                                             dof_values[i + c * dofs_per_comp],
                                             std::integral_constant<bool, true>());
          else
            for (unsigned int v = 0; v < n_lanes; ++v)
              for (unsigned int c = 0; c < n_components; ++c)
                if (indices[v] != dealii::numbers::invalid_unsigned_int)
                  vec.local_element(indices[v] + offset * n_components + c) +=
                    dof_values[i + c * dofs_per_comp][v];
          ++i;

          if (i1 == 0 || i1 == fe_degree - 1)
            {
              ++compressed_i1;
              offset_i1 = 0;
            }
          else
            ++offset_i1;
        }
      if (i2 == 0 || i2 == fe_degree - 1)
        {
          ++compressed_i2;
          offset_i2 = 0;
        }
      else
        ++offset_i2;
    }
}


#endif
