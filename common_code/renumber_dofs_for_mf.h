#ifndef renumber_dofs_for_mf_h
#define renumber_dofs_for_mf_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <unordered_set>

template <int dim, typename Number, typename VectorizedArrayType>
class Renumber
{
public:
  Renumber(const unsigned int a, const unsigned int r, const unsigned int g)
    : assembly_strat(a)
    , renumber_strat(r)
    , grouping_strat(g)
  {}

  void
  renumber(dealii::DoFHandler<dim> &                dof_handler,
           const dealii::AffineConstraints<double> &constraints,
           const typename dealii::MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
             &mf_data) const;

  std::string
  get_renumber_string() const;

private:
  /* helper functions that handle the different strategy types */
  std::vector<unsigned int>
  get_assembly_result(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const;
  std::function<void(std::vector<unsigned int> &,
                     unsigned int &,
                     std::unordered_set<dealii::types::global_dof_index>,
                     const bool &,
                     const unsigned int &)>
  get_renumber_func() const;
  std::vector<unsigned char>
  get_grouping_result(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const;

  /* assembly strategies */
  std::vector<unsigned int>
  cell_assembly(const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const;
  std::vector<unsigned int>
  cellbatch_assembly(const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const;

  /* renumber strategies */
  static void
  first_touch_renumber(std::vector<unsigned int> &numbers_mf_order,
                       unsigned int &             counter_dof_numbers,
                       std::unordered_set<dealii::types::global_dof_index> /*set_dofs*/,
                       const bool &        is_element,
                       const unsigned int &index_within_set);
  static void
  last_touch_renumber(std::vector<unsigned int> &                         numbers_mf_order,
                      unsigned int &                                      counter_dof_numbers,
                      std::unordered_set<dealii::types::global_dof_index> set_dofs,
                      const bool &                                        is_element,
                      const unsigned int &                                index_within_set);

  std::vector<unsigned int>
  grouping(const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
           const std::vector<unsigned int> &                           numbers_mf_order) const;

  void
  base_grouping(std::vector<unsigned int> &      new_numbers,
                const std::vector<unsigned int> &single_domain_dofs,
                const std::vector<unsigned int> &numbers_mf_order) const;

  void
  touch_count_grouping(std::vector<unsigned int> &                                 new_numbers,
                       const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
                       const std::vector<unsigned int> &single_domain_dofs,
                       const std::vector<unsigned int> &numbers_mf_order) const;

  /* grouping strategies */
  std::vector<unsigned char>
  touch_count_cellbatch(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const;
  std::vector<unsigned char>
  touch_count_cellbatch_range(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const;

  /// retrieve the dof - processors (domain) mapping
  std::map<
    std::vector<unsigned int>,
    std::vector<unsigned int>,
    std::function<bool(const std::vector<unsigned int> &, const std::vector<unsigned int> &)>>
  domain_dof_mapping(const dealii::DoFHandler<dim> &dof_handler) const;

  const unsigned int assembly_strat;
  const unsigned int renumber_strat;
  const unsigned int grouping_strat;
};

template <int dim, typename Number, typename VectorizedArrayType>
void
Renumber<dim, Number, VectorizedArrayType>::renumber(
  dealii::DoFHandler<dim> &                                                            dof_handler,
  const dealii::AffineConstraints<double> &                                            constraints,
  const typename dealii::MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData &mf_data)
  const
{
  // do nothing, if base strat selected
  if (renumber_strat == 0)
    return;

  typename dealii::MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData my_mf_data =
    mf_data;
  my_mf_data.initialize_mapping = false;
  dealii::MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(dof_handler,
                     constraints,
                     dealii::QGauss<1>(dof_handler.get_fe().degree + 1),
                     my_mf_data);

  auto numbers_mf_order = get_assembly_result(matrix_free);
  AssertThrow(numbers_mf_order.size() == dof_handler.locally_owned_dofs().n_elements(),
              dealii::ExcMessage("Expected " +
                                 std::to_string(dof_handler.locally_owned_dofs().n_elements()) +
                                 " dofs, touched " + std::to_string(numbers_mf_order.size())));

  auto new_numbers = grouping(matrix_free, numbers_mf_order);
  AssertThrow(new_numbers.size() == dof_handler.locally_owned_dofs().n_elements(),
              dealii::ExcMessage("Dimension mismatch " + std::to_string(new_numbers.size()) +
                                 " vs " +
                                 std::to_string(dof_handler.locally_owned_dofs().n_elements())));

  // list of new indices for each dof, enumerated in the same order as current dofs
  std::vector<dealii::types::global_dof_index> new_global_numbers(
    dof_handler.locally_owned_dofs().n_elements());
  for (unsigned int i = 0; i < new_numbers.size(); ++i)
    new_global_numbers[new_numbers[i]] = dof_handler.locally_owned_dofs().nth_index_in_set(i);

  dof_handler.renumber_dofs(new_global_numbers);
}

template <int dim, typename Number, typename VectorizedArrayType>
std::string
Renumber<dim, Number, VectorizedArrayType>::get_renumber_string() const
{
  std::stringstream ss;
  switch (assembly_strat)
    {
      case 0:
        ss << "cell";
        break;
      case 1:
        ss << "cellbatch";
        break;
      default:
        assert(false);
    }
  ss << "-";
  switch (renumber_strat)
    {
      case 0:
        ss << "base";
        break;
      case 1:
        ss << "first";
        break;
      case 2:
        ss << "last";
        break;
      default:
        assert(false);
    }
  ss << "-";
  switch (grouping_strat)
    {
      case 0:
        ss << "base";
        break;
      case 1:
        ss << "cellbatch";
        break;
      case 2:
        ss << "cellbatch_range";
        break;
      default:
        assert(false);
    }
  return ss.str();
}

template <int dim, typename Number, typename VectorizedArrayType>
std::vector<unsigned int>
Renumber<dim, Number, VectorizedArrayType>::get_assembly_result(
  const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const
{
  switch (assembly_strat)
    {
      case 0:
        return cell_assembly(matrix_free);
      case 1:
        return cellbatch_assembly(matrix_free);
      default:
        assert(false);
    }
}

template <int dim, typename Number, typename VectorizedArrayType>
std::function<void(std::vector<unsigned int> &,
                   unsigned int &,
                   std::unordered_set<dealii::types::global_dof_index>,
                   const bool &,
                   const unsigned int &)>
Renumber<dim, Number, VectorizedArrayType>::get_renumber_func() const
{
  switch (renumber_strat)
    {
      case 1:
        return Renumber::first_touch_renumber;
      case 2:
        return Renumber::last_touch_renumber;
      default:
        assert(false);
    }
}

template <int dim, typename Number, typename VectorizedArrayType>
std::vector<unsigned char>
Renumber<dim, Number, VectorizedArrayType>::get_grouping_result(
  const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const
{
  switch (grouping_strat)
    {
      case 1:
        return touch_count_cellbatch(matrix_free);
      case 2:
        return touch_count_cellbatch_range(matrix_free);
      default:
        assert(false);
    }
}

template <int dim, typename Number, typename VectorizedArrayType>
std::vector<unsigned int>
Renumber<dim, Number, VectorizedArrayType>::cell_assembly(
  const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const
{
  const auto &dof_handler = matrix_free.get_dof_handler();
  const auto &local_dofs  = dof_handler.locally_owned_dofs();

  std::vector<unsigned int> numbers_mf_order(local_dofs.n_elements(),
                                             dealii::numbers::invalid_unsigned_int);
  // contains the dof indices of a cell
  std::vector<dealii::types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);
  // counter depicting the order that is incremented after each dof is assigned a value
  unsigned int       counter_dof_numbers = 0;
  const unsigned int fe_degree           = dof_handler.get_fe().degree;

  auto renumber_func = get_renumber_func();

  for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
    {
      for (unsigned int cell = 0; cell < matrix_free.n_components_filled(cell_batch); ++cell)
        {
          // stores the indices for the dofs in cell_batch
          matrix_free.get_cell_iterator(cell_batch, cell)->get_dof_indices(dof_indices);

          // inclusion of dof_index indicates a previous value assignment within this element
          // only for last_touch_renumber!!
          std::unordered_set<dealii::types::global_dof_index> set_dofs;
          set_dofs.reserve(matrix_free.n_components_filled(cell_batch));

          unsigned int cf = 0;
          for (; cf < dealii::GeometryInfo<dim>::vertices_per_cell; ++cf)
            renumber_func(numbers_mf_order,
                          counter_dof_numbers,
                          set_dofs,
                          local_dofs.is_element(dof_indices[cf]),
                          local_dofs.index_within_set(dof_indices[cf]));

          for (unsigned int line = 0; line < dealii::GeometryInfo<dim>::lines_per_cell; ++line)
            {
              for (unsigned int i = 0; i < fe_degree - 1; ++i, ++cf)
                renumber_func(numbers_mf_order,
                              counter_dof_numbers,
                              set_dofs,
                              local_dofs.is_element(dof_indices[cf]),
                              local_dofs.index_within_set(dof_indices[cf]));
            }
          for (unsigned int quad = 0; quad < dealii::GeometryInfo<dim>::quads_per_cell; ++quad)
            {
              for (unsigned int i = 0; i < (fe_degree - 1) * (fe_degree - 1); ++i, ++cf)
                renumber_func(numbers_mf_order,
                              counter_dof_numbers,
                              set_dofs,
                              local_dofs.is_element(dof_indices[cf]),
                              local_dofs.index_within_set(dof_indices[cf]));
            }
          for (unsigned int hex = 0; hex < dealii::GeometryInfo<dim>::hexes_per_cell; ++hex)
            {
              for (unsigned int i = 0; i < (fe_degree - 1) * (fe_degree - 1) * (fe_degree - 1);
                   ++i, ++cf)
                renumber_func(numbers_mf_order,
                              counter_dof_numbers,
                              set_dofs,
                              local_dofs.is_element(dof_indices[cf]),
                              local_dofs.index_within_set(dof_indices[cf]));
            }
        }
    }
  return numbers_mf_order;
}

template <int dim, typename Number, typename VectorizedArrayType>
std::vector<unsigned int>
Renumber<dim, Number, VectorizedArrayType>::cellbatch_assembly(
  const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const
{
  const auto &dof_handler = matrix_free.get_dof_handler();
  const auto &local_dofs  = dof_handler.locally_owned_dofs();

  std::vector<unsigned int> numbers_mf_order(local_dofs.n_elements(),
                                             dealii::numbers::invalid_unsigned_int);
  // contains the dof indices of a cell
  std::vector<dealii::types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);

  // counter depicting the order that is incremented after each dof is assigned a value
  unsigned int       counter_dof_numbers = 0;
  const unsigned int fe_degree           = dof_handler.get_fe().degree;

  auto renumber_func = get_renumber_func();
  for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
    {
      // inclusion of dof_index indicates a previous value assignment within this element
      // only for last_touch_renumber!!
      std::unordered_set<dealii::types::global_dof_index> set_dofs;
      set_dofs.reserve(matrix_free.n_components_filled(cell_batch) *
                       dof_handler.get_fe().dofs_per_cell);

      // matrix_free.get_dof_info().get_dof_indices_on_cell_batch(dof_indices, cell_batch, true);
      unsigned int cf = 0;
      for (; cf < dealii::GeometryInfo<dim>::vertices_per_cell; ++cf)
        {
          for (unsigned int cell = 0; cell < matrix_free.n_components_filled(cell_batch); ++cell)
            {
              matrix_free.get_cell_iterator(cell_batch, cell)->get_dof_indices(dof_indices);
              renumber_func(numbers_mf_order,
                            counter_dof_numbers,
                            set_dofs,
                            local_dofs.is_element(dof_indices[cf]),
                            local_dofs.index_within_set(dof_indices[cf]));
            }
        }
      for (unsigned int line = 0; line < dealii::GeometryInfo<dim>::lines_per_cell; ++line)
        {
          for (unsigned int i = 0; i < fe_degree - 1; ++i, ++cf)
            {
              for (unsigned int cell = 0; cell < matrix_free.n_components_filled(cell_batch);
                   ++cell)
                {
                  matrix_free.get_cell_iterator(cell_batch, cell)->get_dof_indices(dof_indices);
                  renumber_func(numbers_mf_order,
                                counter_dof_numbers,
                                set_dofs,
                                local_dofs.is_element(dof_indices[cf]),
                                local_dofs.index_within_set(dof_indices[cf]));
                }
            }
        }
      for (unsigned int quad = 0; quad < dealii::GeometryInfo<dim>::quads_per_cell; ++quad)
        {
          for (unsigned int i = 0; i < (fe_degree - 1) * (fe_degree - 1); ++i, ++cf)
            {
              for (unsigned int cell = 0; cell < matrix_free.n_components_filled(cell_batch);
                   ++cell)
                {
                  matrix_free.get_cell_iterator(cell_batch, cell)->get_dof_indices(dof_indices);
                  renumber_func(numbers_mf_order,
                                counter_dof_numbers,
                                set_dofs,
                                local_dofs.is_element(dof_indices[cf]),
                                local_dofs.index_within_set(dof_indices[cf]));
                }
            }
        }
      for (unsigned int hex = 0; hex < dealii::GeometryInfo<dim>::hexes_per_cell; ++hex)
        {
          for (unsigned int i = 0; i < (fe_degree - 1) * (fe_degree - 1) * (fe_degree - 1);
               ++i, ++cf)
            {
              for (unsigned int cell = 0; cell < matrix_free.n_components_filled(cell_batch);
                   ++cell)
                {
                  matrix_free.get_cell_iterator(cell_batch, cell)->get_dof_indices(dof_indices);
                  renumber_func(numbers_mf_order,
                                counter_dof_numbers,
                                set_dofs,
                                local_dofs.is_element(dof_indices[cf]),
                                local_dofs.index_within_set(dof_indices[cf]));
                }
            }
        }
    }
  return numbers_mf_order;
}

template <int dim, typename Number, typename VectorizedArrayType>
void
Renumber<dim, Number, VectorizedArrayType>::first_touch_renumber(
  std::vector<unsigned int> &numbers_mf_order,
  unsigned int &             counter_dof_numbers,
  std::unordered_set<dealii::types::global_dof_index> /*set_dofs*/,
  const bool &        is_element,
  const unsigned int &index_within_set)
{
  if (is_element && numbers_mf_order[index_within_set] == dealii::numbers::invalid_unsigned_int)
    {
      numbers_mf_order[index_within_set] = counter_dof_numbers++;
    }
}

template <int dim, typename Number, typename VectorizedArrayType>
void
Renumber<dim, Number, VectorizedArrayType>::last_touch_renumber(
  std::vector<unsigned int> &                         numbers_mf_order,
  unsigned int &                                      counter_dof_numbers,
  std::unordered_set<dealii::types::global_dof_index> set_dofs,
  const bool &                                        is_element,
  const unsigned int &                                index_within_set)
{
  if (is_element && set_dofs.find(index_within_set) == set_dofs.end())
    {
      numbers_mf_order[index_within_set] = counter_dof_numbers++;
      set_dofs.emplace(index_within_set);
    }
}

template <int dim, typename Number, typename VectorizedArrayType>
std::vector<unsigned int>
Renumber<dim, Number, VectorizedArrayType>::grouping(
  const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
  const std::vector<unsigned int> &                           numbers_mf_order) const
{
  const auto &dof_handler = matrix_free.get_dof_handler();

  auto sorted_entries = domain_dof_mapping(dof_handler);

  // retrieve all dofs touched only by a single processor (non-ghost dofs)
  const std::vector<unsigned int> single_domain_dofs = sorted_entries[std::vector<unsigned int>()];
  sorted_entries.erase(std::vector<unsigned int>());

  // list of dofs defining the new processing order
  std::vector<unsigned int> new_numbers;
  new_numbers.reserve(dof_handler.locally_owned_dofs().n_elements());

  if (grouping_strat == 0)
    base_grouping(new_numbers, single_domain_dofs, numbers_mf_order);
  else
    touch_count_grouping(new_numbers, matrix_free, single_domain_dofs, numbers_mf_order);

  const unsigned int fill_size = new_numbers.size();

  // add all dofs that are touched by multiple processors (=ghost dofs)
  for (const auto &[multi_domains, ghost_dofs] : sorted_entries)
    for (const auto dof : ghost_dofs)
      new_numbers.push_back(dof);

  // renumber within the chunks to reflect the numbering we want to impose
  // through the matrix-free loop
  auto comp = [&](const unsigned int a, const unsigned int b) {
    return (numbers_mf_order[a] < numbers_mf_order[b]);
  };
  std::sort(new_numbers.begin() + fill_size, new_numbers.end(), comp);

  return new_numbers;
}

template <int dim, typename Number, typename VectorizedArrayType>
void
Renumber<dim, Number, VectorizedArrayType>::base_grouping(
  std::vector<unsigned int> &      new_numbers,
  const std::vector<unsigned int> &single_domain_dofs,
  const std::vector<unsigned int> &numbers_mf_order) const
{
  // add all dofs that are touched once
  for (auto i : single_domain_dofs)
    new_numbers.push_back(i);

  // renumber within the chunks to reflect the numbering we want to impose
  // through the matrix-free loop
  auto comp = [&](const unsigned int a, const unsigned int b) {
    return (numbers_mf_order[a] < numbers_mf_order[b]);
  };
  std::sort(new_numbers.begin(), new_numbers.end(), comp);
}

template <int dim, typename Number, typename VectorizedArrayType>
void
Renumber<dim, Number, VectorizedArrayType>::touch_count_grouping(
  std::vector<unsigned int> &                                 new_numbers,
  const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
  const std::vector<unsigned int> &                           single_domain_dofs,
  const std::vector<unsigned int> &                           numbers_mf_order) const
{
  const auto touch_count = get_grouping_result(matrix_free);

  // add all dofs that are touched once
  for (auto i : single_domain_dofs)
    if (touch_count[i] == 1)
      new_numbers.push_back(i);

  // renumber within the chunks to reflect the numbering we want to impose
  // through the matrix-free loop
  auto comp = [&](const unsigned int a, const unsigned int b) {
    return (numbers_mf_order[a] < numbers_mf_order[b]);
  };
  std::sort(new_numbers.begin(), new_numbers.end(), comp);
  const unsigned int fill_size = new_numbers.size();

  // add all dofs that are touched multiple times
  for (auto i : single_domain_dofs)
    if (touch_count[i] > 1)
      new_numbers.push_back(i);

  std::sort(new_numbers.begin() + fill_size, new_numbers.end(), comp);
}

template <int dim, typename Number, typename VectorizedArrayType>
std::vector<unsigned char>
Renumber<dim, Number, VectorizedArrayType>::touch_count_cellbatch(
  const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const
{
  const auto n_dofs = matrix_free.get_dof_handler().locally_owned_dofs().n_elements();

  std::vector<unsigned char> touch_count(n_dofs, 0);

  for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
    {
      // contains local dof indices for a cell batch
      std::vector<unsigned int> local_dof_indices;

      // retrieve the local dof indices for this cell_batch batch
      matrix_free.get_dof_info().get_dof_indices_on_cell_batch(local_dof_indices, cell_batch, true);

      // sort & remove duplicates
      std::sort(local_dof_indices.begin(), local_dof_indices.end());
      local_dof_indices.erase(std::unique(local_dof_indices.begin(), local_dof_indices.end()),
                              local_dof_indices.end());

      // calc touch_count
      for (unsigned int i : local_dof_indices)
        if (i < n_dofs)
          touch_count[i]++;
    }
  return touch_count;
}

template <int dim, typename Number, typename VectorizedArrayType>
std::vector<unsigned char>
Renumber<dim, Number, VectorizedArrayType>::touch_count_cellbatch_range(
  const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const
{
  const auto n_dofs = matrix_free.get_dof_handler().locally_owned_dofs().n_elements();

  std::vector<unsigned char> touch_count(n_dofs, 0);

  // required information to identify cellbatch_ranges
  const auto &partition_row_index = matrix_free.get_task_info().partition_row_index;
  const auto &cell_partition_data = matrix_free.get_task_info().cell_partition_data;

  for (auto part = 0u; part < partition_row_index.size() - 2; ++part)
    {
      // cellbatch_range loop
      for (auto i = partition_row_index[part]; i < partition_row_index[part + 1]; ++i)
        {
          // contains local dof indices for a cellbatch_range
          std::vector<unsigned int> cellbatch_range_indices;

          for (unsigned int cellbatch = cell_partition_data[i];
               cellbatch < cell_partition_data[i + 1];
               ++cellbatch)
            {
              // contains local dof indices for a cellbatch
              std::vector<unsigned int> cellbatch_indices;
              // retrieve the local dof indices for this cell_batch batch
              matrix_free.get_dof_info().get_dof_indices_on_cell_batch(cellbatch_indices,
                                                                       cellbatch,
                                                                       true);
              cellbatch_range_indices.insert(cellbatch_range_indices.end(),
                                             cellbatch_indices.begin(),
                                             cellbatch_indices.end());
            }

          // sort & remove duplicates
          std::sort(cellbatch_range_indices.begin(), cellbatch_range_indices.end());
          cellbatch_range_indices.erase(std::unique(cellbatch_range_indices.begin(),
                                                    cellbatch_range_indices.end()),
                                        cellbatch_range_indices.end());

          // calc touch_count
          for (unsigned int i : cellbatch_range_indices)
            if (i < n_dofs)
              touch_count[i]++;
        }
    }
  return touch_count;
}

template <int dim, typename Number, typename VectorizedArrayType>
std::map<std::vector<unsigned int>,
         std::vector<unsigned int>,
         std::function<bool(const std::vector<unsigned int> &, const std::vector<unsigned int> &)>>
Renumber<dim, Number, VectorizedArrayType>::domain_dof_mapping(
  const dealii::DoFHandler<dim> &dof_handler) const
{
  // contains the ranks the dof is involved with
  std::vector<std::vector<unsigned int>> processors_involved(
    dof_handler.locally_owned_dofs().n_elements());
  // contains the dof indices of a cell
  std::vector<dealii::types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);
  // retrieves the involved ranks for each dof
  for (auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_ghost())
      {
        cell->get_dof_indices(dof_indices);
        for (auto i : dof_indices)
          {
            if (dof_handler.locally_owned_dofs().is_element(i))
              processors_involved[dof_handler.locally_owned_dofs().index_within_set(i)].push_back(
                cell->subdomain_id());
          }
      }
  // sort & remove duplicates
  for (std::vector<unsigned int> &v : processors_involved)
    if (v.size() > 1)
      {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
      }

  // calc (vector of processors , the dofs these processors are involved with ) - pairs
  std::map<
    std::vector<unsigned int>,
    std::vector<unsigned int>,
    std::function<bool(const std::vector<unsigned int> &, const std::vector<unsigned int> &)>>
    sorted_entries{[](const std::vector<unsigned int> &a, const std::vector<unsigned int> &b) {
      if (a.size() < b.size())
        return true;
      if (a.size() == b.size())
        {
          for (unsigned int i = 0; i < a.size(); ++i)
            if (a[i] < b[i])
              return true;
            else if (a[i] > b[i])
              return false;
        }
      return false;
    }};
  for (unsigned int i = 0; i < processors_involved.size(); ++i)
    {
      std::vector<unsigned int> &v = sorted_entries[processors_involved[i]];
      v.push_back(i);
    }

  return sorted_entries;
}

#endif
