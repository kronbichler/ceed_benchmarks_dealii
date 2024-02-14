
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>

#include "../common_code/kershaw.h"
#include "../common_code/poisson_operator.h"


using namespace dealii;

const unsigned int dimension      = 3;
const bool         do_p_multigrid = true;
const bool         do_coarse_amg  = true;
using MyQuadrature                = QGaussLobatto<1>;



template <typename T>
int
sign(T val)
{
  return (T(0) < val) - (val < T(0));
}

template <int dim>
class SolutionFunction : public Function<dim>
{
  static constexpr unsigned int k = 2;

public:
  double
  value(const Point<dim> &p, const unsigned int = 0) const override
  {
    double result = 1.;
    for (unsigned int d = 0; d < dim; ++d)
      {
        const double sk = std::sin(2 * k * numbers::PI * p[d]);
        if (sk != -0.)
          result *= std::exp(-1. / (sk * sk)) * sign(sk);
        else
          result = 0.;
      }
    return result;
  }

  double
  laplacian(const Point<dim> &p, const unsigned int = 0) const override
  {
    std::array<double, 3> val, lapl;
    for (unsigned int d = 0; d < dim; ++d)
      {
        const double sk  = std::sin(2 * k * numbers::PI * p[d]);
        const double skp = 2 * k * numbers::PI * std::cos(2 * k * numbers::PI * p[d]);
        const double skpp =
          -Utilities::fixed_power<2>(2 * k * numbers::PI) * std::sin(2 * k * numbers::PI * p[d]);
        if (sk != -0.)
          {
            val[d]  = std::exp(-1. / (sk * sk)) * sign(sk);
            lapl[d] = val[d] *
                      ((4. / (sk * sk * sk * sk * sk * sk) - 6. / (sk * sk * sk * sk)) * skp * skp +
                       2. / (sk * sk * sk) * skpp);
          }
        else
          val[d] = lapl[d] = 0.;
      }
    double result = 0.;
    for (unsigned int d = 0; d < dim; ++d)
      {
        double tmp = lapl[d];
        for (unsigned int e = 0; e < dim; ++e)
          if (d != e)
            tmp *= val[e];
        result += tmp;
      }
    return result;
  }
};



template <int dim, typename Number = double>
class LaplaceOperatorEVector : public Poisson::LaplaceOperator<dim, 1, Number>
{
public:
  /**
   * Number typedef.
   */
  using value_type = Number;

  /**
   * size_type needed for preconditioner classes.
   */
  using size_type = types::global_dof_index;

  /**
   * size_type needed for preconditioner classes.
   */
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  /**
   * Make number of components available as variable
   */
  static constexpr unsigned int n_components = 1;

  using BaseClass = Poisson::LaplaceOperator<dim, 1, Number>;

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    Timer                                 time;
    std::pair<unsigned int, unsigned int> cell_range(0, this->data->n_cell_batches());
    EXPAND_OPERATIONS(local_apply_evector);

    Poisson::internal::set_constrained_entries(this->data->get_constrained_dofs(0), src, dst);
    continuous_vector.compress(VectorOperation::add);
    continuous_vector.update_ghost_values();
    FEEvaluation<dim, -1, 0, n_components, Number> phi_c(*this->data, 0);
    FEEvaluation<dim, -1, 0, n_components, Number> phi_d(*this->data, 1);
    for (unsigned int cell = 0; cell < this->data->n_cell_batches(); ++cell)
      {
        phi_d.reinit(cell);
        phi_c.reinit(cell);
        phi_c.read_dof_values(continuous_vector);
        for (unsigned int i = 0; i < phi_c.dofs_per_cell; ++i)
          phi_d.begin_dof_values()[i] = phi_c.begin_dof_values()[i];
        phi_d.set_dof_values(dst);
      }
    this->compute_time += time.wall_time();
  }

  void
  vmult_add(VectorType &dst, const VectorType &src) const
  {
    std::pair<unsigned int, unsigned int> cell_range(0, this->data->n_cell_batches());
    EXPAND_OPERATIONS(local_apply_evector);

    Poisson::internal::set_constrained_entries(this->data->get_constrained_dofs(0), src, dst);
    continuous_vector.compress(VectorOperation::add);
    continuous_vector.update_ghost_values();
    FEEvaluation<dim, -1, 0, n_components, Number> phi_c(*this->data, 0);
    FEEvaluation<dim, -1, 0, n_components, Number> phi_d(*this->data, 1);
    for (unsigned int cell = 0; cell < this->data->n_cell_batches(); ++cell)
      {
        phi_c.reinit(cell);
        phi_c.read_dof_values(continuous_vector);
        for (unsigned int i = 0; i < phi_c.dofs_per_cell; ++i)
          phi_d.begin_dof_values()[i] = phi_c.begin_dof_values()[i];
        phi_d.reinit(cell);
        phi_d.distribute_local_to_global(dst);
      }
  }

  LinearAlgebra::distributed::Vector<Number>
  compute_inverse_diagonal() const
  {
    LinearAlgebra::distributed::Vector<Number> diag = BaseClass::compute_inverse_diagonal();
    diag.update_ghost_values();
    LinearAlgebra::distributed::Vector<Number> dg_diagonal;
    this->data->initialize_dof_vector(dg_diagonal, 1);
    FEEvaluation<dim, -1, 0, n_components, Number> phi_c(*this->data, 0);
    FEEvaluation<dim, -1, 0, n_components, Number> phi_d(*this->data, 1);
    for (unsigned int cell = 0; cell < this->data->n_cell_batches(); ++cell)
      {
        phi_d.reinit(cell);
        phi_c.reinit(cell);
        phi_c.read_dof_values(diag);
        for (unsigned int i = 0; i < phi_c.dofs_per_cell; ++i)
          phi_d.begin_dof_values()[i] = phi_c.begin_dof_values()[i];
        phi_d.set_dof_values(dg_diagonal);
      }
    return dg_diagonal;
  }

  types::global_dof_index
  m() const
  {
    return this->data->get_dof_handler(1).n_dofs();
  }

private:
  template <int fe_degree, int n_q_points_1d>
  void
  local_apply_evector(VectorType &,
                      const VectorType &                           src,
                      const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    if (!continuous_vector.get_partitioner()->is_compatible(
          *this->data->get_dof_info(0).vector_partitioner))
      this->initialize_dof_vector(continuous_vector);
    continuous_vector = 0.;
    FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> phi_c(*this->data, 0);
    FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> phi_d(*this->data, 1);
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_c.reinit(cell);
        phi_d.reinit(cell);
        phi_d.read_dof_values(src);
        this->template apply_local_cell<n_q_points_1d>(phi_d,
                                                       cell,
                                                       phi_d.begin_dof_values(),
                                                       phi_c.begin_dof_values());
        phi_c.distribute_local_to_global(continuous_vector);
      }
  }
  mutable VectorType continuous_vector;
};



template <typename number>
class MGCoarseSolverAMG : public MGCoarseGridBase<LinearAlgebra::distributed::Vector<number>>
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;

  // use better name for PETSc's internal vector
  using VectorTypePETSc = Vec;

  MGCoarseSolverAMG(const PETScWrappers::MPI::SparseMatrix &system_matrix,
                    const unsigned int                      n_v_cycles)
    : system_matrix(system_matrix)
  {
    PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
    data.symmetric_operator = true;
    using RelaxationType    = PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType;
    data.relaxation_type_up = RelaxationType::Chebyshev;               // symmetricSORJacobi;
    data.relaxation_type_down             = RelaxationType::Chebyshev; // symmetricSORJacobi;
    data.max_iter                         = n_v_cycles;
    data.aggressive_coarsening_num_levels = 0;
    data.output_details                   = true;
    algebraic_multigrid.initialize(system_matrix, data);
    const IndexSet owned_elements = system_matrix.locally_owned_range_indices();
    VecCreateMPI(system_matrix.get_mpi_communicator(),
                 owned_elements.n_elements(),
                 PETSC_DETERMINE,
                 &petsc_dst);
    VecCreateMPI(system_matrix.get_mpi_communicator(),
                 owned_elements.n_elements(),
                 PETSC_DETERMINE,
                 &petsc_src);
    compute_time = 0;
  }

  double
  get_compute_time_and_reset()
  {
    const auto   comm = MPI_COMM_WORLD;
    const double result =
      Utilities::MPI::sum(compute_time, comm) / Utilities::MPI::n_mpi_processes(comm);
    compute_time = 0.;
    return result;
  }

  ~MGCoarseSolverAMG()
  {
    if (false && system_matrix.m() > 0)
      {
        PetscErrorCode ierr = VecDestroy(&petsc_dst);
        AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
        ierr = VecDestroy(&petsc_src);
        AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));

        const auto         min_max_avg = Utilities::MPI::min_max_avg(compute_time, MPI_COMM_WORLD);
        const unsigned int n_digits_ranks =
          Utilities::needed_digits(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1);
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << " Time " << std::left << std::setw(21) << "solve coarse" << std::setw(11)
                    << min_max_avg.min << " [p" << std::setw(n_digits_ranks)
                    << min_max_avg.min_index << "] " << std::setw(11) << min_max_avg.avg
                    << std::setw(11) << min_max_avg.max << " [p" << std::setw(n_digits_ranks)
                    << min_max_avg.max_index << "]" << std::endl;
      }
  }

  void
  operator()(const unsigned int level, VectorType &dst, const VectorType &src) const override
  {
    Timer time;
    AssertThrow(level == 0, ExcNotImplemented());
    {
      // copy to PETSc internal vector type
      PetscInt       begin, end;
      PetscErrorCode ierr = VecGetOwnershipRange(petsc_src, &begin, &end);
      AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));

      PetscScalar *ptr;
      ierr = VecGetArray(petsc_src, &ptr);
      AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));

      const PetscInt local_size = src.get_partitioner()->locally_owned_size();
      AssertDimension(local_size, static_cast<unsigned int>(end - begin));
      for (PetscInt i = 0; i < local_size; ++i)
        ptr[i] = src.local_element(i);

      ierr = VecRestoreArray(petsc_src, &ptr);
      AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
    }

    // wrap `Vec` (aka VectorTypePETSc) into VectorBase (without copying data)
    dealii::PETScWrappers::VectorBase vector_dst(petsc_dst);
    dealii::PETScWrappers::VectorBase vector_src(petsc_src);

    algebraic_multigrid.vmult(vector_dst, vector_src);

    {
      PetscInt       begin, end;
      PetscErrorCode ierr = VecGetOwnershipRange(petsc_dst, &begin, &end);
      AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));

      PetscScalar *ptr;
      ierr = VecGetArray(petsc_dst, &ptr);
      AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));

      const PetscInt local_size = dst.get_partitioner()->locally_owned_size();
      AssertDimension(local_size, static_cast<unsigned int>(end - begin));

      for (PetscInt i = 0; i < local_size; ++i)
        dst.local_element(i) = ptr[i];

      ierr = VecRestoreArray(petsc_dst, &ptr);
      AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
    }
    compute_time += time.wall_time();
  }

private:
  const PETScWrappers::MPI::SparseMatrix &system_matrix;
  PETScWrappers::PreconditionBoomerAMG    algebraic_multigrid;
  mutable VectorTypePETSc                 petsc_src;
  mutable VectorTypePETSc                 petsc_dst;
  mutable double                          compute_time;
};



template <int dim>
class LaplaceProblem
{
public:
  LaplaceProblem(const unsigned int degree);

  void
  run(const unsigned int n_refinements, const double epsy, const double epsz);

private:
  void
  setup_grid(const unsigned int n_refinements, const double epsy, const double epsz);
  void
  setup_dofs();
  void
  setup_matrix_free();
  void
  assemble_rhs();
  void
  setup_transfer();
  void
  setup_smoother();
  void
  setup_coarse_solver();
  void
  solve();
  void
  compute_error() const;

  parallel::distributed::Triangulation<dim> triangulation;

  FE_Q<dim>       fe;
  FE_DGQ<dim>     fe_dg;
  DoFHandler<dim> dof_handler;
  DoFHandler<dim> dof_handler_dg;

  std::vector<unsigned int>      p_mg_level_degrees;
  MGLevelObject<DoFHandler<dim>> level_dof_handlers;
  MGLevelObject<DoFHandler<dim>> level_dof_handlers_dg;

  MappingQCache<dim> mapping;

  AffineConstraints<double> constraints;
  using SystemMatrixType = LaplaceOperatorEVector<dim, double>;
  SystemMatrixType system_matrix;

  MGLevelObject<AffineConstraints<float>> level_constraints;
  MGConstrainedDoFs                       mg_constrained_dofs;
  using LevelMatrixType = LaplaceOperatorEVector<dim, float>;
  using VectorType      = typename LevelMatrixType::VectorType;
  MGLevelObject<LevelMatrixType> mg_matrices;

  LinearAlgebra::distributed::Vector<double> solution;
  LinearAlgebra::distributed::Vector<double> system_rhs;

  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>>           mg_transfers_p;
  std::unique_ptr<MGTransferGlobalCoarsening<dim, VectorType>> mg_transfer_p;
  std::unique_ptr<MGCoarseGridBase<VectorType>>                mg_coarse;

  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
  mg::SmootherRelaxation<SmootherType, VectorType> mg_smoother;

  ConditionalOStream pcout;
};



template <int dim>
LaplaceProblem<dim>::LaplaceProblem(const unsigned int degree)
#ifdef DEAL_II_WITH_P4EST
  : triangulation(MPI_COMM_WORLD,
                  Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
#else
  : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
  , fe(degree)
  , fe_dg(degree)
  , dof_handler(triangulation)
  , dof_handler_dg(triangulation)
  , mapping(1)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{}



template <int dim>
void
LaplaceProblem<dim>::setup_grid(const unsigned int n_refinements,
                                const double       epsy,
                                const double       epsz)
{
  const bool refine_before_deformation = true;
  GridGenerator::subdivided_hyper_cube(triangulation, 5);

  const auto transformation_function = [epsy, epsz](const Point<dim> &in_point) {
    Point<3> temp_point;
    kershaw(epsy,
            epsz,
            in_point[0],
            in_point[1],
            dim == 3 ? in_point[2] : 0.0,
            temp_point[0],
            temp_point[1],
            temp_point[2]);

    Point<dim> out_point;
    for (unsigned int i = 0; i < dim; ++i)
      out_point[i] = temp_point[i];
    return out_point;
  };

  if (refine_before_deformation)
    {
      triangulation.refine_global(n_refinements);
      const auto transform =
        [transformation_function](const typename Triangulation<dim>::cell_iterator &,
                                  const Point<dim> &in_point) {
          return transformation_function(in_point);
        };
      mapping.initialize(MappingQ1<dim>(), triangulation, transform, false);
    }
  else
    {
      GridTools::transform(transformation_function, triangulation);
      triangulation.refine_global(n_refinements);
      mapping.initialize(MappingQ1<dim>(), triangulation);
    }
}



template <int dim>
void
LaplaceProblem<dim>::setup_dofs()
{
  system_matrix.clear();
  mg_matrices.clear_elements();

  dof_handler.distribute_dofs(fe);
  dof_handler_dg.distribute_dofs(fe_dg);

  pcout << " Number of DoFs: " << dof_handler_dg.n_dofs() << std::endl;

  typename MatrixFree<dim, float>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme             = MatrixFree<dim, float>::AdditionalData::none;
  additional_data.overlap_communication_computation = false;

  // Create p-hierarchy
  if (do_p_multigrid)
    {
      AssertThrow(fe.degree > 1, ExcNotImplemented());
      unsigned int degree = fe.degree;
      p_mg_level_degrees.clear();
      while (degree > 1)
        {
          degree -= 2;
          p_mg_level_degrees.push_back(std::max(1U, degree));
        }
      while (degree > 1)
        {
          degree -= 1;
          p_mg_level_degrees.push_back(degree);
        }
      pcout << " p-multigrid coarsening strategy: " << fe.degree;
      for (const unsigned int i : p_mg_level_degrees)
        pcout << " " << i;
      pcout << std::endl;
      std::reverse(p_mg_level_degrees.begin(), p_mg_level_degrees.end());

      const unsigned int max_level = p_mg_level_degrees.size();
      level_dof_handlers.resize(0, max_level - 1);
      level_dof_handlers_dg.resize(0, max_level - 1);
      for (unsigned int level = 0; level < max_level; ++level)
        {
          level_dof_handlers[level].reinit(triangulation);
          level_dof_handlers[level].distribute_dofs(FE_Q<dim>(p_mg_level_degrees[level]));
          level_dof_handlers_dg[level].reinit(triangulation);
          level_dof_handlers_dg[level].distribute_dofs(FE_DGQ<dim>(p_mg_level_degrees[level]));
        }
    }

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();
}



template <int dim>
void
LaplaceProblem<dim>::setup_matrix_free()
{
  std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(new MatrixFree<dim, double>());
  {
    AffineConstraints<double> empty_constraints;
    empty_constraints.close();
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points);
    std::vector<const DoFHandler<dim> *>           dof_handlers{&dof_handler, &dof_handler_dg};
    std::vector<const AffineConstraints<double> *> all_constraints{&constraints,
                                                                   &empty_constraints};
    system_mf_storage->reinit(
      mapping, dof_handlers, all_constraints, MyQuadrature(fe.degree + 1), additional_data);
  }

  const bool analyze_mesh = false;
  if (analyze_mesh)
    {
      double max_jac = 0., min_jac = std::numeric_limits<double>::max(), avg = 0,
             min_gll = std::numeric_limits<double>::max(), max_gll = 0;
      const unsigned int    n_dofs_1d = fe.degree + 1;
      FEEvaluation<dim, -1> eval(*system_mf_storage);
      unsigned int          count      = 0;
      unsigned int          strides[3] = {1, n_dofs_1d, n_dofs_1d * n_dofs_1d};
      for (unsigned int cell = 0; cell < system_mf_storage->n_cell_batches(); ++cell)
        {
          eval.reinit(cell);
          for (unsigned int q = 0; q < eval.n_q_points; ++q)
            for (unsigned int d = 0; d < system_mf_storage->n_active_entries_per_cell_batch(cell);
                 ++d)
              {
                const auto jdet = eval.JxW(q) / system_mf_storage->get_mapping_info()
                                                  .cell_data[0]
                                                  .descriptor[0]
                                                  .quadrature_weights[q];
                max_jac = std::max(jdet[d], max_jac);
                min_jac = std::min(jdet[d], min_jac);
                avg += jdet[d];
                ++count;
              }
          for (unsigned int d = 0; d < system_mf_storage->n_active_entries_per_cell_batch(cell);
               ++d)
            for (unsigned int e = 0; e < dim; ++e)
              {
                min_gll =
                  std::min(min_gll,
                           eval.quadrature_point(strides[e]).distance(eval.quadrature_point(0))[d]);
                min_gll =
                  std::min(min_gll,
                           eval.quadrature_point(strides[e] * (n_dofs_1d - 1))
                             .distance(eval.quadrature_point(strides[e] * (n_dofs_1d - 2)))[d]);
                max_gll =
                  std::max(max_gll,
                           eval.quadrature_point(strides[e] * (n_dofs_1d / 2))
                             .distance(eval.quadrature_point(strides[e] * (n_dofs_1d / 2 + 1)))[d]);
              }
        }
      avg /= count;
      std::cout << "Scaled Jacobians: " << min_jac / max_jac << " " << avg / max_jac << " jac "
                << max_jac << std::endl;
      std::cout << "Min/max GLL: " << min_gll << " " << max_gll << std::endl;
    }

  system_matrix.initialize(system_mf_storage);

  system_mf_storage->initialize_dof_vector(solution, 1);
  system_mf_storage->initialize_dof_vector(system_rhs, 1);
  using LevelNumber = typename LevelMatrixType::value_type;

  if (do_p_multigrid)
    {
      const unsigned int max_level = p_mg_level_degrees.size();
      level_constraints.resize(0, max_level);
      mg_matrices.resize(0, max_level);
      for (unsigned int level = 0; level <= max_level; ++level)
        {
          const DoFHandler<dim> &dof_h =
            level == max_level ? dof_handler : level_dof_handlers[level];
          const DoFHandler<dim> &dof_h_dg =
            level == max_level ? dof_handler_dg : level_dof_handlers_dg[level];
          IndexSet locally_relevant_dofs;
          DoFTools::extract_locally_relevant_dofs(dof_h, locally_relevant_dofs);

          level_constraints[level].clear();
          level_constraints[level].reinit(locally_relevant_dofs);
          AffineConstraints<double> constraint;
          constraint.reinit(locally_relevant_dofs);
          DoFTools::make_hanging_node_constraints(dof_h, constraint);
          VectorTools::interpolate_boundary_values(
            mapping, dof_h, 0, Functions::ZeroFunction<dim>(), constraint);
          constraint.close();
          level_constraints[level].copy_from(constraint);
          AffineConstraints<LevelNumber> empty_constraints;
          empty_constraints.close();
          std::vector<const DoFHandler<dim> *>                dof_handlers{&dof_h, &dof_h_dg};
          std::vector<const AffineConstraints<LevelNumber> *> all_constraints{
            &level_constraints[level], &empty_constraints};

          typename MatrixFree<dim, LevelNumber>::AdditionalData additional_data;
          additional_data.tasks_parallel_scheme =
            MatrixFree<dim, LevelNumber>::AdditionalData::none;
          additional_data.mapping_update_flags =
            (update_gradients | update_JxW_values | update_quadrature_points);
          auto mg_mf_storage_level = std::make_shared<MatrixFree<dim, LevelNumber>>();
          mg_mf_storage_level->reinit(mapping,
                                      dof_handlers,
                                      all_constraints,
                                      MyQuadrature(dof_h.get_fe().degree + 1),
                                      additional_data);

          mg_matrices[level].initialize(mg_mf_storage_level);
        }
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }
}



template <int dim>
void
LaplaceProblem<dim>::assemble_rhs()
{
  Timer time;

  LinearAlgebra::distributed::Vector<double> rhs_continuous;
  system_matrix.initialize_dof_vector(rhs_continuous);
  FEEvaluation<dim, -1> phi(*system_matrix.get_matrix_free());
  SolutionFunction<dim> exact_solution;
  for (unsigned int cell = 0; cell < system_matrix.get_matrix_free()->n_cell_batches(); ++cell)
    {
      phi.reinit(cell);
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const Point<dim, VectorizedArray<double>> p_vec = phi.quadrature_point(q);
          VectorizedArray<double>                   rhs;
          for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
            {
              Point<dim> p;
              for (unsigned int d = 0; d < dim; ++d)
                p[d] = p_vec[d][v];
              rhs[v] = -exact_solution.laplacian(p);
            }
          phi.submit_value(rhs, q);
        }
      phi.integrate_scatter(EvaluationFlags::values, rhs_continuous);
    }
  rhs_continuous.compress(VectorOperation::add);
  rhs_continuous.update_ghost_values();
  FEEvaluation<dim, -1> phi_d(*system_matrix.get_matrix_free(), 1);
  for (unsigned int cell = 0; cell < system_matrix.get_matrix_free()->n_cell_batches(); ++cell)
    {
      phi_d.reinit(cell);
      phi.reinit(cell);
      phi.read_dof_values(rhs_continuous);
      for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
        phi_d.begin_dof_values()[i] = phi.begin_dof_values()[i];
      phi_d.set_dof_values(system_rhs);
    }
}



template <int dim>
void
LaplaceProblem<dim>::setup_transfer()
{
  if (do_p_multigrid)
    {
      const unsigned int                                      max_level = p_mg_level_degrees.size();
      AffineConstraints<typename LevelMatrixType::value_type> empty_constraints;
      empty_constraints.close();
      mg_transfers_p.resize(0, max_level);
      for (unsigned int level = 0; level < max_level; ++level)
        mg_transfers_p[level + 1].reinit(level + 1 == max_level ? dof_handler_dg :
                                                                  level_dof_handlers_dg[level + 1],
                                         level == 0 ? level_dof_handlers[level] :
                                                      level_dof_handlers_dg[level],
                                         empty_constraints,
                                         level == 0 ? level_constraints[level] : empty_constraints);
      mg_transfer_p = std::make_unique<MGTransferGlobalCoarsening<dim, VectorType>>(
        mg_transfers_p, [&](const unsigned int level, VectorType &vec) {
          mg_matrices[level].get_matrix_free()->initialize_dof_vector(vec, level > 0);
        });
    }
}



template <int dim>
void
LaplaceProblem<dim>::setup_smoother()
{
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, mg_matrices.max_level());
  for (unsigned int level = 0; level <= mg_matrices.max_level(); ++level)
    {
      smoother_data[level].polynomial_type =
        SmootherType::AdditionalData::PolynomialType::first_kind;
      if (level > 0)
        {
          smoother_data[level].smoothing_range     = 11.;
          smoother_data[level].degree              = 3;
          smoother_data[level].eig_cg_n_iterations = 12;
        }
      else if (!do_coarse_amg)
        {
          smoother_data[0].smoothing_range     = 1e-3;
          smoother_data[0].degree              = numbers::invalid_unsigned_int;
          smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
        }
      smoother_data[level].preconditioner = std::make_shared<DiagonalMatrix<VectorType>>();
      smoother_data[level].preconditioner->get_vector() =
        mg_matrices[level].compute_inverse_diagonal();
    }
  mg_smoother.initialize(mg_matrices, smoother_data);
}



template <int dim>
void
LaplaceProblem<dim>::setup_coarse_solver()
{
  if (do_coarse_amg)
    {
      mg_coarse = std::make_unique<MGCoarseSolverAMG<typename LevelMatrixType::value_type>>(
        mg_matrices[0].get_system_matrix(), 1);
    }
  else
    {
      auto coarse_solver = std::make_unique<MGCoarseGridApplySmoother<VectorType>>();
      coarse_solver->initialize(mg_smoother);
      mg_coarse = std::move(coarse_solver);
    }
}



// Wrap multigrid preconditioner classes to measure run time
template <typename MGType>
struct MyPreconditioner
{
  MyPreconditioner(const MGType &preconditioner)
    : preconditioner(preconditioner)
    , compute_time(0.)
  {}

  template <typename VectorType>
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    Timer time;
    preconditioner.vmult(dst, src);
    compute_time += time.wall_time();
  }

  const MGType & preconditioner;
  mutable double compute_time;
};


template <int dim>
void
LaplaceProblem<dim>::solve()
{
  mg::Matrix<VectorType> mg_matrix(mg_matrices);

  IterationNumberControl solver_control(100, 1e-8 * system_rhs.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
  constraints.set_zero(solution);
  double prec_time = 0.;

  Timer time;
  if (do_p_multigrid)
    {
      Multigrid<VectorType> mg(mg_matrix, *mg_coarse, *mg_transfer_p, mg_smoother, mg_smoother);
      PreconditionMG<dim, VectorType, MGTransferGlobalCoarsening<dim, VectorType>> preconditioner(
        dof_handler, mg, *mg_transfer_p);
      MyPreconditioner<PreconditionMG<dim, VectorType, MGTransferGlobalCoarsening<dim, VectorType>>>
        precondition(preconditioner);

      cg.solve(system_matrix, solution, system_rhs, precondition);
      prec_time = precondition.compute_time;
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }
  const double solve_time = time.wall_time();
  pcout << std::setprecision(4);
  pcout << "BPS5 p=" << fe.degree << " E=" << triangulation.n_global_active_cells() << std::endl;
  pcout << "solve time: " << solve_time << "s" << std::endl;
  pcout << "  preconditioner " << prec_time << "s" << std::endl;
  pcout << "    smoother time per level: ";
  for (unsigned int l = mg_matrices.max_level(); l != mg_matrices.min_level(); --l)
    {
      pcout << mg_matrices[l].get_compute_time_and_reset() << "s ";
    }
  pcout << std::endl;
  if (auto coarse =
        dynamic_cast<MGCoarseSolverAMG<typename LevelMatrixType::value_type> *>(mg_coarse.get()))
    pcout << "    coarse grid " << coarse->get_compute_time_and_reset() << "s" << std::endl;
  pcout << "iterations: " << solver_control.last_step() << std::endl;
  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  pcout << "throughput: "
        << dof_handler.n_dofs() * solver_control.last_step() / solve_time / n_ranks
        << " (DOF x iter)/s/rank" << std::endl;
  pcout << "throughput: " << dof_handler.n_dofs() / solve_time / n_ranks << " DOF/s/rank"
        << std::endl
        << std::endl;
}



template <int dim>
void
LaplaceProblem<dim>::compute_error() const
{
  FEValues<dim> fe_values(mapping,
                          fe_dg,
                          QGauss<dim>(fe.degree + 2),
                          update_values | update_JxW_values | update_quadrature_points);
  double        error = 0.;
  solution.update_ghost_values();
  std::vector<double>   solution_values(fe_values.n_quadrature_points);
  SolutionFunction<dim> exact_solution;
  for (const auto &cell : dof_handler_dg.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(solution, solution_values);
        double local_error = 0.;
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          local_error +=
            Utilities::fixed_power<2>(exact_solution.value(fe_values.quadrature_point(q)) -
                                      solution_values[q]) *
            fe_values.JxW(q);
        error += local_error;
      }
  error = Utilities::MPI::sum(error, triangulation.get_communicator());
  pcout << " L2 error of solution: " << std::sqrt(error) << std::endl;
}



template <int dim>
void
LaplaceProblem<dim>::run(const unsigned int n_refinements, const double epsy, const double epsz)
{
  pcout << " Running Poisson problem in " << dim << "D with " << n_refinements
        << " refinements and degree " << fe.degree << std::endl;
  pcout << " Discretization parameters: degree=" << fe.degree << " epsy=" << epsy
        << " epsz=" << epsz << std::endl;
  const unsigned int n_vect_doubles = VectorizedArray<double>::size();
  const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

  pcout << " " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
        << " processes, vectorization over " << n_vect_doubles << " doubles = " << n_vect_bits
        << " bits (" << Utilities::System::get_current_vectorization_level() << ')' << std::endl;
  std::map<std::string, dealii::Timer> timer;

  timer["setup_grid"].start();
  setup_grid(n_refinements, epsy, epsz);
  timer["setup_grid"].stop();

  timer["setup_dofs"].start();
  setup_dofs();
  timer["setup_dofs"].stop();

  timer["setup_matrix_free"].start();
  setup_matrix_free();
  timer["setup_matrix_free"].stop();

  timer["assemble_rhs"].start();
  assemble_rhs();
  timer["assemble_rhs"].stop();

  timer["setup_transfer"].start();
  setup_transfer();
  timer["setup_transfer"].stop();

  timer["setup_smoother"].start();
  setup_smoother();
  timer["setup_smoother"].stop();

  timer["setup_coarse"].start();
  setup_coarse_solver();
  timer["setup_coarse"].stop();

  timer["solve"].start();
  solve();
  timer["solve"].stop();
  solution = 0.;
  timer["solve2"].start();
  solve();
  timer["solve2"].stop();
  solution = 0.;
  timer["solve3"].start();
  solve();
  timer["solve3"].stop();
  solution = 0.;
  timer["solve4"].start();
  solve();
  timer["solve4"].stop();
  solution = 0.;
  timer["solve5"].start();
  solve();
  timer["solve5"].stop();

  const unsigned int n_repeat = 50;
  timer["matvec_double"].start();
  for (unsigned int t = 0; t < n_repeat; ++t)
    system_matrix.vmult(system_rhs, solution);
  timer["matvec_double"].stop();

  using LevelNumber = typename LevelMatrixType::value_type;
  LinearAlgebra::distributed::Vector<LevelNumber> vec1, vec2;
  mg_matrices[mg_matrices.max_level()].get_matrix_free()->initialize_dof_vector(vec1, 1);
  vec2.reinit(vec1);
  timer["matvec_float"].start();
  for (unsigned int t = 0; t < n_repeat; ++t)
    mg_matrices[mg_matrices.max_level()].vmult(vec2, vec1);
  timer["matvec_float"].stop();

  mg_matrices[mg_matrices.max_level() - 1].get_matrix_free()->initialize_dof_vector(vec1, 1);
  vec2.reinit(vec1);
  timer["matvec_float_coarser"].start();
  for (unsigned int t = 0; t < n_repeat; ++t)
    mg_matrices[mg_matrices.max_level() - 1].vmult(vec2, vec1);
  timer["matvec_float_coarser"].stop();

  timer["output"].start();
  if (triangulation.n_global_active_cells() < 10000)
    {
      DataOut<dim> data_out;

      solution.update_ghost_values();
      data_out.attach_dof_handler(dof_handler_dg);
      data_out.add_data_vector(solution, "solution");
      LinearAlgebra::distributed::Vector<double> exact(solution);
      VectorTools::interpolate(mapping, dof_handler_dg, SolutionFunction<dim>(), exact);
      data_out.add_data_vector(exact, "exact");
      Vector<double> aspect_ratios =
        GridTools::compute_aspect_ratio_of_cells(mapping, triangulation, QGauss<dim>(3));
      data_out.add_data_vector(aspect_ratios, "aspect_ratio");

      data_out.build_patches(mapping, fe.degree);

      DataOutBase::VtkFlags flags;
      flags.compression_level        = DataOutBase::CompressionLevel::best_speed;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);
      data_out.write_vtu_with_pvtu_record("./", "solution", n_refinements, MPI_COMM_WORLD, 1);
    }
  timer["output"].stop();

  timer["compute_error"].start();
  compute_error();
  timer["compute_error"].stop();

  pcout << std::endl;
  for (const auto &entry : timer)
    {
      const auto min_max_avg =
        Utilities::MPI::min_max_avg(entry.second.wall_time(), MPI_COMM_WORLD);
      const unsigned int n_digits_ranks =
        Utilities::needed_digits(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1);
      pcout << " Time " << std::left << std::setw(21) << entry.first << std::setw(11)
            << min_max_avg.min << " [p" << std::setw(n_digits_ranks) << min_max_avg.min_index
            << "] " << std::setw(11) << min_max_avg.avg << std::setw(11) << min_max_avg.max << " [p"
            << std::setw(n_digits_ranks) << min_max_avg.max_index << "]" << std::endl;
    }
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  unsigned int degree = 3;
  if (argc > 1)
    degree = std::atoi(argv[1]);

  unsigned int n_refinements = 2;
  if (argc > 2)
    n_refinements = std::atoi(argv[2]);

  double epsy = 1.;
  if (argc > 3)
    epsy = std::atof(argv[3]);

  double epsz = epsy;
  if (argc > 4)
    epsz = std::atof(argv[4]);

  LaplaceProblem<dimension> problem(degree);
  problem.run(n_refinements, epsy, epsz);
}
