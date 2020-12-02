
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
  const unsigned int *   indices =
    compressed_indices.data() + cell_no * n_lanes * dealii::Utilities::pow(3, dim);
  const unsigned char *unconstrained =
    all_indices_unconstrained.data() + cell_no * dealii::Utilities::pow(3, dim);
  constexpr unsigned int dofs_per_comp = dealii::Utilities::pow(fe_degree + 1, dim);
  dealii::internal::VectorReader<Number, VectorizedArrayType> reader;
  // vertex dofs
  for (unsigned int i2 = 0; i2 < (dim == 3 ? 2 : 1); ++i2)
    for (unsigned int i1 = 0; i1 < 2; ++i1)
      for (unsigned int i0 = 0; i0 < 2; ++i0, indices += n_lanes, ++unconstrained)
        for (unsigned int c = 0; c < n_components; ++c)
          if (*unconstrained)
            reader.process_dof_gather(
              indices,
              vec,
              c,
              dof_values[i2 * fe_degree * (fe_degree + 1) * (fe_degree + 1) +
                         i1 * fe_degree * (fe_degree + 1) + i0 * fe_degree + c * dofs_per_comp],
              std::integral_constant<bool, true>());
          else
            for (unsigned int v = 0; v < n_lanes; ++v)
              dof_values[i2 * fe_degree * (fe_degree + 1) * (fe_degree + 1) +
                         i1 * fe_degree * (fe_degree + 1) + i0 * fe_degree + c * dofs_per_comp][v] =
                (indices[v] == dealii::numbers::invalid_unsigned_int) ?
                  0. :
                  vec.local_element(indices[v] + c);

  // line dofs
  constexpr unsigned int offsets[2][4] = {
    {fe_degree + 1, 2 * fe_degree + 1, 1, (fe_degree + 1) * fe_degree + 1},
    {fe_degree * (fe_degree + 1) * (fe_degree + 1) + fe_degree + 1,
     fe_degree * (fe_degree + 1) * (fe_degree + 1) + 2 * fe_degree + 1,
     fe_degree * (fe_degree + 1) * (fe_degree + 1) + 1,
     (fe_degree + 1) * (fe_degree + 1) * (fe_degree + 1) - fe_degree}};
  for (unsigned int d = 0; d < dim - 1; ++d)
    {
      for (unsigned int l = 0; l < 2; ++l)
        {
          if (*unconstrained)
            for (unsigned int i = 0; i < fe_degree - 1; ++i)
              for (unsigned int c = 0; c < n_components; ++c)
                reader.process_dof_gather(
                  indices,
                  vec,
                  i * n_components + c,
                  dof_values[offsets[d][l] + i * (fe_degree + 1) + c * dofs_per_comp],
                  std::integral_constant<bool, true>());
          else
            for (unsigned int i = 0; i < fe_degree - 1; ++i)
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    dof_values[offsets[d][l] + i * (fe_degree + 1) + c * dofs_per_comp][v] =
                      vec.local_element(indices[v] + i * n_components + c);
                  else
                    dof_values[offsets[d][l] + i * (fe_degree + 1) + c * dofs_per_comp][v] = 0.;
          indices += n_lanes;
          ++unconstrained;
        }
      for (unsigned int l = 2; l < 4; ++l)
        {
          if (*unconstrained)
            {
              for (unsigned int i = 0; i < fe_degree - 1; ++i)
                for (unsigned int c = 0; c < n_components; ++c)
                  reader.process_dof_gather(indices,
                                            vec,
                                            i * n_components + c,
                                            dof_values[offsets[d][l] + i + c * dofs_per_comp],
                                            std::integral_constant<bool, true>());
            }
          else
            for (unsigned int i = 0; i < fe_degree - 1; ++i)
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    dof_values[offsets[d][l] + i + c * dofs_per_comp][v] =
                      vec.local_element(indices[v] + i * n_components + c);
                  else
                    dof_values[offsets[d][l] + i + c * dofs_per_comp][v] = 0;
          indices += n_lanes;
          ++unconstrained;
        }
    }
  if (dim == 3)
    {
      constexpr unsigned int strides2    = (fe_degree + 1) * (fe_degree + 1);
      constexpr unsigned int offsets2[4] = {strides2,
                                            strides2 + fe_degree,
                                            strides2 + (fe_degree + 1) * fe_degree,
                                            strides2 + (fe_degree + 1) * (fe_degree + 1) - 1};
      for (unsigned int l = 0; l < 4; ++l)
        {
          if (*unconstrained)
            for (unsigned int i = 0; i < fe_degree - 1; ++i)
              for (unsigned int c = 0; c < n_components; ++c)
                reader.process_dof_gather(
                  indices,
                  vec,
                  i * n_components + c,
                  dof_values[offsets2[l] + i * strides2 + c * dofs_per_comp],
                  std::integral_constant<bool, true>());
          else
            for (unsigned int i = 0; i < fe_degree - 1; ++i)
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    dof_values[offsets2[l] + i * strides2 + c * dofs_per_comp][v] =
                      vec.local_element(indices[v] + i * n_components + c);
                  else
                    dof_values[offsets2[l] + i * strides2 + c * dofs_per_comp][v] = 0.;
          indices += n_lanes;
          ++unconstrained;
        }

      // face dofs
      for (unsigned int face = 0; face < 6; ++face)
        {
          const unsigned int stride1 = dealii::Utilities::pow(fe_degree + 1, (face / 2 + 1) % dim);
          const unsigned int stride2 = dealii::Utilities::pow(fe_degree + 1, (face / 2 + 2) % dim);
          const unsigned int offset =
            ((face % 2 == 0) ? 0 : fe_degree) * dealii::Utilities::pow(fe_degree + 1, face / 2);
          if (*unconstrained)
            for (unsigned int i2 = 1, j = 0; i2 < fe_degree; ++i2)
              for (unsigned int i1 = 1; i1 < fe_degree; ++i1, ++j)
                for (unsigned int c = 0; c < n_components; ++c)
                  reader.process_dof_gather(
                    indices,
                    vec,
                    j * n_components + c,
                    dof_values[offset + i2 * stride2 + i1 * stride1 + c * dofs_per_comp],
                    std::integral_constant<bool, true>());
          else
            for (unsigned int i2 = 1, j = 0; i2 < fe_degree; ++i2)
              for (unsigned int i1 = 1; i1 < fe_degree; ++i1, ++j)
                for (unsigned int v = 0; v < n_lanes; ++v)
                  for (unsigned int c = 0; c < n_components; ++c)
                    if (indices[v] != dealii::numbers::invalid_unsigned_int)
                      dof_values[offset + i2 * stride2 + i1 * stride1 + c * dofs_per_comp][v] =
                        vec.local_element(indices[v] + j * n_components + c);
                    else
                      dof_values[offset + i2 * stride2 + i1 * stride1 + c * dofs_per_comp][v] = 0;
          indices += n_lanes;
          ++unconstrained;
        }
    }

  // cell dofs
  if (*unconstrained)
    for (unsigned int i2 = 1, j = 0; i2 < (dim == 3 ? fe_degree : 2); ++i2)
      for (unsigned int i1 = 1; i1 < fe_degree; ++i1)
        {
          for (unsigned int i0 = 1; i0 < fe_degree; ++i0, ++j)
            for (unsigned int c = 0; c < n_components; ++c)
              reader.process_dof_gather(indices,
                                        vec,
                                        j * n_components + c,
                                        dof_values[i2 * (fe_degree + 1) * (fe_degree + 1) +
                                                   i1 * (fe_degree + 1) + i0 + c * dofs_per_comp],
                                        std::integral_constant<bool, true>());
        }
  else
    for (unsigned int i2 = 1, j = 0; i2 < (dim == 3 ? fe_degree : 2); ++i2)
      for (unsigned int i1 = 1; i1 < fe_degree; ++i1)
        for (unsigned int i0 = 1; i0 < fe_degree; ++i0, ++j)
          for (unsigned int v = 0; v < n_lanes; ++v)
            for (unsigned int c = 0; c < n_components; ++c)
              if (indices[v] != dealii::numbers::invalid_unsigned_int)
                dof_values[i2 * (fe_degree + 1) * (fe_degree + 1) + i1 * (fe_degree + 1) + i0 +
                           c * dofs_per_comp][v] =
                  vec.local_element(indices[v] + j * n_components + c);
              else
                dof_values[i2 * (fe_degree + 1) * (fe_degree + 1) + i1 * (fe_degree + 1) + i0 +
                           c * dofs_per_comp][v] = 0;
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
  const unsigned int *   indices =
    compressed_indices.data() + cell_no * n_lanes * dealii::Utilities::pow(3, dim);
  const unsigned char *unconstrained =
    all_indices_unconstrained.data() + cell_no * dealii::Utilities::pow(3, dim);
  constexpr unsigned int dofs_per_comp = dealii::Utilities::pow(fe_degree + 1, dim);
  dealii::internal::VectorDistributorLocalToGlobal<Number, VectorizedArrayType> distributor;

  // vertex dofs
  for (unsigned int i2 = 0; i2 < (dim == 3 ? 2 : 1); ++i2)
    for (unsigned int i1 = 0; i1 < 2; ++i1)
      for (unsigned int i0 = 0; i0 < 2; ++i0, indices += n_lanes, ++unconstrained)
        for (unsigned int c = 0; c < n_components; ++c)
          if (*unconstrained)
            distributor.process_dof_gather(
              indices,
              vec,
              c,
              dof_values[i2 * fe_degree * (fe_degree + 1) * (fe_degree + 1) +
                         i1 * fe_degree * (fe_degree + 1) + i0 * fe_degree + c * dofs_per_comp],
              std::integral_constant<bool, true>());
          else
            for (unsigned int v = 0; v < n_lanes; ++v)
              if (indices[v] != dealii::numbers::invalid_unsigned_int)
                vec.local_element(indices[v] + c) +=
                  dof_values[i2 * fe_degree * (fe_degree + 1) * (fe_degree + 1) +
                             i1 * fe_degree * (fe_degree + 1) + i0 * fe_degree + c * dofs_per_comp]
                            [v];

  // line dofs
  constexpr unsigned int offsets[2][4] = {
    {fe_degree + 1, 2 * fe_degree + 1, 1, (fe_degree + 1) * fe_degree + 1},
    {fe_degree * (fe_degree + 1) * (fe_degree + 1) + fe_degree + 1,
     fe_degree * (fe_degree + 1) * (fe_degree + 1) + 2 * fe_degree + 1,
     fe_degree * (fe_degree + 1) * (fe_degree + 1) + 1,
     (fe_degree + 1) * (fe_degree + 1) * (fe_degree + 1) - fe_degree}};
  for (unsigned int d = 0; d < dim - 1; ++d)
    {
      for (unsigned int l = 0; l < 2; ++l)
        {
          if (*unconstrained)
            for (unsigned int i = 0; i < fe_degree - 1; ++i)
              for (unsigned int c = 0; c < n_components; ++c)
                distributor.process_dof_gather(
                  indices,
                  vec,
                  i * n_components + c,
                  dof_values[offsets[d][l] + i * (fe_degree + 1) + c * dofs_per_comp],
                  std::integral_constant<bool, true>());
          else
            for (unsigned int i = 0; i < fe_degree - 1; ++i)
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    vec.local_element(indices[v] + i * n_components + c) +=
                      dof_values[offsets[d][l] + i * (fe_degree + 1) + c * dofs_per_comp][v];
          indices += n_lanes;
          ++unconstrained;
        }
      for (unsigned int l = 2; l < 4; ++l)
        {
          if (*unconstrained)
            {
              for (unsigned int i = 0; i < fe_degree - 1; ++i)
                for (unsigned int c = 0; c < n_components; ++c)
                  distributor.process_dof_gather(indices,
                                                 vec,
                                                 i * n_components + c,
                                                 dof_values[offsets[d][l] + i + c * dofs_per_comp],
                                                 std::integral_constant<bool, true>());
            }
          else
            for (unsigned int i = 0; i < fe_degree - 1; ++i)
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    vec.local_element(indices[v] + i * n_components + c) +=
                      dof_values[offsets[d][l] + i + c * dofs_per_comp][v];
          indices += n_lanes;
          ++unconstrained;
        }
    }
  if (dim == 3)
    {
      constexpr unsigned int strides2    = (fe_degree + 1) * (fe_degree + 1);
      constexpr unsigned int offsets2[4] = {strides2,
                                            strides2 + fe_degree,
                                            strides2 + (fe_degree + 1) * fe_degree,
                                            strides2 + (fe_degree + 1) * (fe_degree + 1) - 1};
      for (unsigned int l = 0; l < 4; ++l)
        {
          if (*unconstrained)
            for (unsigned int i = 0; i < fe_degree - 1; ++i)
              for (unsigned int c = 0; c < n_components; ++c)
                distributor.process_dof_gather(
                  indices,
                  vec,
                  i * n_components + c,
                  dof_values[offsets2[l] + i * strides2 + c * dofs_per_comp],
                  std::integral_constant<bool, true>());
          else
            for (unsigned int i = 0; i < fe_degree - 1; ++i)
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    vec.local_element(indices[v] + i * n_components + c) +=
                      dof_values[offsets2[l] + i * strides2 + c * dofs_per_comp][v];
          indices += n_lanes;
          ++unconstrained;
        }

      // face dofs
      for (unsigned int face = 0; face < 6; ++face)
        {
          const unsigned int stride1 = dealii::Utilities::pow(fe_degree + 1, (face / 2 + 1) % dim);
          const unsigned int stride2 = dealii::Utilities::pow(fe_degree + 1, (face / 2 + 2) % dim);
          const unsigned int offset =
            ((face % 2 == 0) ? 0 : fe_degree) * dealii::Utilities::pow(fe_degree + 1, face / 2);
          if (*unconstrained)
            for (unsigned int i2 = 1, j = 0; i2 < fe_degree; ++i2)
              for (unsigned int i1 = 1; i1 < fe_degree; ++i1, ++j)
                for (unsigned int c = 0; c < n_components; ++c)
                  distributor.process_dof_gather(
                    indices,
                    vec,
                    j * n_components + c,
                    dof_values[offset + i2 * stride2 + i1 * stride1 + c * dofs_per_comp],
                    std::integral_constant<bool, true>());
          else
            for (unsigned int i2 = 1, j = 0; i2 < fe_degree; ++i2)
              for (unsigned int i1 = 1; i1 < fe_degree; ++i1, ++j)
                for (unsigned int v = 0; v < n_lanes; ++v)
                  for (unsigned int c = 0; c < n_components; ++c)
                    if (indices[v] != dealii::numbers::invalid_unsigned_int)
                      vec.local_element(indices[v] + j * n_components + c) +=
                        dof_values[offset + i2 * stride2 + i1 * stride1 + c * dofs_per_comp][v];
          indices += n_lanes;
          ++unconstrained;
        }
    }

  // cell dofs
  if (*unconstrained)
    for (unsigned int i2 = 1, j = 0; i2 < (dim == 3 ? fe_degree : 2); ++i2)
      for (unsigned int i1 = 1; i1 < fe_degree; ++i1)
        {
          for (unsigned int i0 = 1; i0 < fe_degree; ++i0, ++j)
            for (unsigned int c = 0; c < n_components; ++c)
              distributor.process_dof_gather(
                indices,
                vec,
                j * n_components + c,
                dof_values[i2 * (fe_degree + 1) * (fe_degree + 1) + i1 * (fe_degree + 1) + i0 +
                           c * dofs_per_comp],
                std::integral_constant<bool, true>());
        }
  else
    for (unsigned int i2 = 1, j = 0; i2 < (dim == 3 ? fe_degree : 2); ++i2)
      for (unsigned int i1 = 1; i1 < fe_degree; ++i1)
        for (unsigned int i0 = 1; i0 < fe_degree; ++i0, ++j)
          for (unsigned int v = 0; v < n_lanes; ++v)
            for (unsigned int c = 0; c < n_components; ++c)
              if (indices[v] != dealii::numbers::invalid_unsigned_int)
                vec.local_element(indices[v] + j * n_components + c) +=
                  dof_values[i2 * (fe_degree + 1) * (fe_degree + 1) + i1 * (fe_degree + 1) + i0 +
                             c * dofs_per_comp][v];
}


#endif
