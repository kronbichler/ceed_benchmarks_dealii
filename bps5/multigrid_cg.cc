
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
const bool         compute_metric = true;
using MyQuadrature                = QGaussLobatto<1>;

template <int dim, typename number>
class LaplaceOperator
  : public MatrixFreeOperators::Base<dim,
                                     LinearAlgebra::distributed::Vector<number>>
{
public:
  using value_type = number;
  using VectorType = LinearAlgebra::distributed::Vector<number>;

  LaplaceOperator();

  void
  initialize(std::shared_ptr<const MatrixFree<dim, number>> data)
  {
    this->MatrixFreeOperators::Base<dim, VectorType>::initialize(data);
    compute_vertex_coefficients();
  }

  void
  initialize(std::shared_ptr<const MatrixFree<dim, number>> data,
             const MGConstrainedDoFs &                      mg_constrained_dofs,
             const unsigned int                             level)
  {
    this->MatrixFreeOperators::Base<dim, VectorType>::initialize(
      data, mg_constrained_dofs, level);
    compute_vertex_coefficients();
  }

  void
  vmult(VectorType &dst, const VectorType &src) const;

  void
  vmult(VectorType &      dst,
        const VectorType &src,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_before_loop,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_after_loop) const;

  virtual void
  compute_diagonal() override;

  const PETScWrappers::MPI::SparseMatrix &
  get_system_matrix() const;

private:
  virtual void
  apply_add(VectorType &dst, const VectorType &src) const override;

  void
  local_apply(const MatrixFree<dim, number> &              data,
              VectorType &                                 dst,
              const VectorType &                           src,
              const std::pair<unsigned int, unsigned int> &cell_range) const;

  void
  do_cell_integral_local(FEEvaluation<dim, -1, 0, 1, number> &integrator) const
  {
    integrator.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate(EvaluationFlags::gradients);
  }

  void
  local_compute_diagonal(
    const MatrixFree<dim, number> &              data,
    VectorType &                                 dst,
    const unsigned int &                         dummy,
    const std::pair<unsigned int, unsigned int> &cell_range) const;

  void
  compute_vertex_coefficients();

  mutable PETScWrappers::MPI::SparseMatrix system_matrix;
  AlignedVector<std::array<Tensor<1, dim, VectorizedArray<number>>,
                           GeometryInfo<dim>::vertices_per_cell>>
    cell_vertex_coefficients;
};



template <int dim, typename number>
LaplaceOperator<dim, number>::LaplaceOperator()
  : MatrixFreeOperators::Base<dim, VectorType>()
{}



namespace
{
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
} // namespace



template <int dim, typename number>
void
LaplaceOperator<dim, number>::local_apply(
  const MatrixFree<dim, number> &              data,
  VectorType &                                 dst,
  const VectorType &                           src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  FEEvaluation<dim, -1, 0, 1, number> phi(data);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::gradients);
      if (compute_metric)
        {
          const std::array<Tensor<1, dim, VectorizedArray<number>>,
                           GeometryInfo<dim>::vertices_per_cell> &v =
            cell_vertex_coefficients[cell];
          const unsigned int n_q_points_1d =
            phi.get_shape_info().data[0].n_q_points_1d;
          const unsigned int   n_q_points = phi.n_q_points;
          const Quadrature<1> &quad_1d =
            phi.get_shape_info().data[0].quadrature;
          VectorizedArray<number> *phi_grads = phi.begin_gradients();
          for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
            {
              const number z     = quad_1d.point(qz)[0];
              const auto   x_con = v[1] + z * v[5];
              const auto   y_con = v[2] + z * v[6];
              const auto   y_var = v[4] + z * v[7];
              for (unsigned int qy = 0; qy < n_q_points_1d; ++qy)
                {
                  const number y     = quad_1d.point(qy)[0];
                  const auto   z_con = v[3] + y * v[6];
                  const auto   z_var = v[5] + y * v[7];
                  const number q_weight_tmp =
                    quad_1d.weight(qz) * quad_1d.weight(qy);
                  const auto x_der = x_con + y * y_var;
                  for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                    {
                      const number x = quad_1d.point(qx)[0];
                      Tensor<2, dim, VectorizedArray<number>> jac;
                      jac[0]         = x_der;
                      jac[1]         = y_con + x * y_var;
                      jac[2]         = z_con + x * z_var;
                      const auto det = do_invert(jac);
                      const auto JxW =
                        det * (q_weight_tmp *
                               static_cast<number>(quad_1d.weight(qx)));

                      const std::size_t q =
                        (qz * n_q_points_1d + qy) * n_q_points_1d + qx;

                      VectorizedArray<number> tmp[dim], tmp2[dim];
                      for (unsigned int d = 0; d < dim; ++d)
                        {
                          tmp[d] = jac[d][0] * phi_grads[q] +
                                   jac[d][1] * phi_grads[q + 1 * n_q_points] +
                                   jac[d][2] * phi_grads[q + 2 * n_q_points];
                          tmp[d] *= JxW;
                        }
                      for (unsigned int d = 0; d < dim; ++d)
                        {
                          tmp2[d] = jac[0][d] * tmp[0];
                          for (unsigned int e = 1; e < dim; ++e)
                            tmp2[d] += jac[e][d] * tmp[e];
                        }
                      phi_grads[q]                  = tmp2[0];
                      phi_grads[q + 1 * n_q_points] = tmp2[1];
                      phi_grads[q + 2 * n_q_points] = tmp2[2];
                    }
                }
            }
        }
      else
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_gradient(phi.get_gradient(q), q);
      phi.integrate_scatter(EvaluationFlags::gradients, dst);
    }
}



template <int dim, typename number>
void
LaplaceOperator<dim, number>::apply_add(VectorType &      dst,
                                        const VectorType &src) const
{
  this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src);
}



template <int dim, typename number>
void
LaplaceOperator<dim, number>::vmult(VectorType &      dst,
                                    const VectorType &src) const
{
  this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src, true);
  for (unsigned int i : this->data->get_constrained_dofs())
    dst.local_element(i) = src.local_element(i);
}



template <int dim, typename number>
void
LaplaceOperator<dim, number>::vmult(
  VectorType &      dst,
  const VectorType &src,
  const std::function<void(const unsigned int, const unsigned int)>
    &operation_before_loop,
  const std::function<void(const unsigned int, const unsigned int)>
    &operation_after_loop) const
{
  this->data->cell_loop(&LaplaceOperator::local_apply,
                        this,
                        dst,
                        src,
                        operation_before_loop,
                        operation_after_loop);
  for (unsigned int i : this->data->get_constrained_dofs())
    dst.local_element(i) = src.local_element(i);
}



template <int dim, typename number>
void
LaplaceOperator<dim, number>::compute_diagonal()
{
  this->inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
  VectorType &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(inverse_diagonal);
  unsigned int dummy = 0;
  this->data->cell_loop(&LaplaceOperator::local_compute_diagonal,
                        this,
                        inverse_diagonal,
                        dummy);

  this->set_constrained_entries_to_one(inverse_diagonal);

  for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
    {
      Assert(inverse_diagonal.local_element(i) > 0.,
             ExcMessage("No diagonal entry in a positive definite operator "
                        "should be zero"));
      inverse_diagonal.local_element(i) =
        1. / inverse_diagonal.local_element(i);
    }
}


template <int dim, typename number>
void
LaplaceOperator<dim, number>::compute_vertex_coefficients()
{
  cell_vertex_coefficients.resize(this->data->n_cell_batches());
  for (unsigned int c = 0; c < this->data->n_cell_batches(); ++c)
    {
      for (unsigned int l = 0;
           l < this->data->n_active_entries_per_cell_batch(c);
           ++l)
        {
          const typename DoFHandler<dim>::cell_iterator cell =
            this->data->get_cell_iterator(c, l);
          if (dim == 2)
            {
              std::array<Tensor<1, dim>, 4> v{{cell->vertex(0),
                                               cell->vertex(1),
                                               cell->vertex(2),
                                               cell->vertex(3)}};
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
        }
      for (unsigned int l = this->data->n_active_entries_per_cell_batch(c);
           l < VectorizedArray<number>::size();
           ++l)
        {
          for (unsigned int d = 0; d < dim; ++d)
            cell_vertex_coefficients[c][d + 1][d][l] = 1.;
        }
    }
}


template <int dim, typename number>
void
LaplaceOperator<dim, number>::local_compute_diagonal(
  const MatrixFree<dim, number> &data,
  VectorType &                   dst,
  const unsigned int &,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  FEEvaluation<dim, -1, 0, 1, number> phi(data);

  AlignedVector<VectorizedArray<number>> diagonal(phi.dofs_per_cell);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
            phi.submit_dof_value(VectorizedArray<number>(), j);
          phi.submit_dof_value(make_vectorized_array<number>(1.), i);

          phi.evaluate(EvaluationFlags::gradients);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_gradient(phi.get_gradient(q), q);
          phi.integrate(EvaluationFlags::gradients);
          diagonal[i] = phi.get_dof_value(i);
        }
      for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
        phi.submit_dof_value(diagonal[i], i);
      phi.distribute_local_to_global(dst);
    }
}



template <int dim, typename number>
const PETScWrappers::MPI::SparseMatrix &
LaplaceOperator<dim, number>::get_system_matrix() const
{
  // Check if matrix has already been set up.
  if (system_matrix.m() == 0 && system_matrix.n() == 0)
    {
      // Set up sparsity pattern of system matrix.
      const auto &dof_handler = this->data->get_dof_handler();
      const AffineConstraints<number> &constraints =
        this->data->get_affine_constraints();

      IndexSet           relevant_dofs;
      const unsigned int level = this->data->get_mg_level();
      if (level == numbers::invalid_unsigned_int)
        DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
      else
        DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                      level,
                                                      relevant_dofs);
      DynamicSparsityPattern dsp(relevant_dofs.size(),
                                 relevant_dofs.size(),
                                 relevant_dofs);
      if (level == numbers::invalid_unsigned_int)
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      else
        MGTools::make_sparsity_pattern(dof_handler, dsp, level, constraints);

      const IndexSet &owned_dofs = level == numbers::invalid_unsigned_int ?
                                     dof_handler.locally_owned_dofs() :
                                     dof_handler.locally_owned_mg_dofs(level);
      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 owned_dofs,
                                                 dof_handler.get_communicator(),
                                                 relevant_dofs);

      system_matrix.reinit(owned_dofs,
                           owned_dofs,
                           dsp,
                           dof_handler.get_communicator());

      MatrixFreeTools::compute_matrix(*this->data,
                                      constraints,
                                      system_matrix,
                                      &LaplaceOperator::do_cell_integral_local,
                                      this);
    }

  return this->system_matrix;
}


template <int dim, typename MatrixType>
class MGTransferMF
  : public MGTransferMatrixFree<dim, typename MatrixType::value_type>
{
public:
  void
  setup(const MGLevelObject<MatrixType> &laplace,
        const MGConstrainedDoFs &        mg_constrained_dofs)
  {
    this->MGTransferMatrixFree<dim, typename MatrixType::value_type>::
      initialize_constraints(mg_constrained_dofs);
    laplace_operator = &laplace;
  }

  template <class InVector, int spacedim>
  void
  copy_to_mg(
    const DoFHandler<dim, spacedim> &mg_dof,
    MGLevelObject<
      LinearAlgebra::distributed::Vector<typename MatrixType::value_type>> &dst,
    const InVector &src) const
  {
    for (unsigned int level = dst.min_level(); level <= dst.max_level();
         ++level)
      (*laplace_operator)[level].initialize_dof_vector(dst[level]);
    MGLevelGlobalTransfer<LinearAlgebra::distributed::Vector<
      typename MatrixType::value_type>>::copy_to_mg(mg_dof, dst, src);
  }

private:
  const MGLevelObject<MatrixType> *laplace_operator;
};



template <typename number>
class MGCoarseSolverAMG
  : public MGCoarseGridBase<LinearAlgebra::distributed::Vector<number>>
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
    using RelaxationType =
      PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType;
    data.relaxation_type_up = RelaxationType::Chebyshev; // symmetricSORJacobi;
    data.relaxation_type_down =
      RelaxationType::Chebyshev; // symmetricSORJacobi;
    data.max_iter                         = n_v_cycles;
    data.aggressive_coarsening_num_levels = 1;
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

  ~MGCoarseSolverAMG()
  {
    if (system_matrix.m() > 0)
      {
        PetscErrorCode ierr = VecDestroy(&petsc_dst);
        AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
        ierr = VecDestroy(&petsc_src);
        AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));

        const auto min_max_avg =
          Utilities::MPI::min_max_avg(compute_time, MPI_COMM_WORLD);
        const unsigned int n_digits_ranks = Utilities::needed_digits(
          Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1);
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << " Time " << std::left << std::setw(21) << "solve coarse"
                    << std::setw(11) << min_max_avg.min << " [p"
                    << std::setw(n_digits_ranks) << min_max_avg.min_index
                    << "] " << std::setw(11) << min_max_avg.avg << std::setw(11)
                    << min_max_avg.max << " [p" << std::setw(n_digits_ranks)
                    << min_max_avg.max_index << "]" << std::endl;
      }
  }

  void
  operator()(const unsigned int level,
             VectorType &       dst,
             const VectorType & src) const override
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
  setup_grid(const unsigned int n_refinements,
             const double       epsy,
             const double       epsz);
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

  parallel::distributed::Triangulation<dim> triangulation;

  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;

  std::vector<unsigned int>      p_mg_level_degrees;
  MGLevelObject<DoFHandler<dim>> level_dof_handlers;

  MappingQCache<dim> mapping;

  AffineConstraints<double> constraints;
  using SystemMatrixType = Poisson::LaplaceOperator<dim, 1, double>;
  SystemMatrixType system_matrix;

  MGLevelObject<AffineConstraints<float>> level_constraints;
  MGConstrainedDoFs                       mg_constrained_dofs;
  using LevelMatrixType = Poisson::LaplaceOperator<dim, 1, float>;
  using VectorType      = typename LevelMatrixType::VectorType;
  MGLevelObject<LevelMatrixType> mg_matrices;

  LinearAlgebra::distributed::Vector<double> solution;
  LinearAlgebra::distributed::Vector<double> system_rhs;

  MGTransferMF<dim, LevelMatrixType>                           mg_transfer;
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
  : triangulation(
      MPI_COMM_WORLD,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
#else
  : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
  , fe(degree)
  , dof_handler(triangulation)
  , mapping(1)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{}



template <int dim>
void
LaplaceProblem<dim>::setup_grid(const unsigned int n_refinements,
                                const double       epsy,
                                const double       epsz)
{
  const bool refine_before_deformation = false;
  GridGenerator::subdivided_hyper_cube(triangulation, 6);

  const auto transformation_function = [epsy,
                                        epsz](const Point<dim> &in_point) {
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
        [transformation_function](
          const typename Triangulation<dim>::cell_iterator &,
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

  pcout << " Number of DoFs: " << dof_handler.n_dofs() << std::endl;

  typename MatrixFree<dim, float>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, float>::AdditionalData::none;

  // Renumber DoFs
  if (do_p_multigrid)
    {
      AssertThrow(fe.degree > 1, ExcNotImplemented());
      unsigned int degree = fe.degree;
      p_mg_level_degrees.clear();
      while (degree > 3)
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
      for (unsigned int level = 0; level < max_level; ++level)
        {
          level_dof_handlers[level].reinit(triangulation);
          level_dof_handlers[level].distribute_dofs(
            FE_Q<dim>(p_mg_level_degrees[level]));
        }

      for (unsigned int level = 0; level <= max_level; ++level)
        {
          DoFHandler<dim> &dof_h =
            level == max_level ? dof_handler : level_dof_handlers[level];
          IndexSet relevant_dofs;
          DoFTools::extract_locally_relevant_dofs(dof_h, relevant_dofs);
          AffineConstraints<double> constraint;
          constraint.reinit(relevant_dofs);
          DoFTools::make_hanging_node_constraints(dof_h, constraint);
          VectorTools::interpolate_boundary_values(
            mapping, dof_h, 0, Functions::ZeroFunction<dim>(), constraint);
          constraint.close();

          DoFRenumbering::matrix_free_data_locality(dof_h,
                                                    constraint,
                                                    additional_data);
        }
    }
  else
    {
      dof_handler.distribute_mg_dofs();
      const std::set<types::boundary_id> dirichlet_boundary = {0};
      mg_constrained_dofs.initialize(dof_handler);
      mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                         dirichlet_boundary);

      for (unsigned int level = 0; level < triangulation.n_global_levels();
           ++level)
        {
          IndexSet relevant_dofs;
          DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                        level,
                                                        relevant_dofs);
          AffineConstraints<double> constraint;
          constraint.reinit(relevant_dofs);
          constraint.add_lines(mg_constrained_dofs.get_boundary_indices(level));
          constraint.close();
          additional_data.mg_level = level;

          DoFRenumbering::matrix_free_data_locality(dof_handler,
                                                    constraint,
                                                    additional_data);
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
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, double>::AdditionalData::none;
  additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points);
  std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(
    new MatrixFree<dim, double>());
  system_mf_storage->reinit(mapping,
                            dof_handler,
                            constraints,
                            MyQuadrature(fe.degree + 1),
                            additional_data);
  system_matrix.initialize(system_mf_storage);

  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(system_rhs);
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

          typename MatrixFree<dim, LevelNumber>::AdditionalData additional_data;
          additional_data.tasks_parallel_scheme =
            MatrixFree<dim, LevelNumber>::AdditionalData::none;
          additional_data.mapping_update_flags =
            (update_gradients | update_JxW_values | update_quadrature_points);
          auto mg_mf_storage_level =
            std::make_shared<MatrixFree<dim, LevelNumber>>();
          mg_mf_storage_level->reinit(mapping,
                                      dof_h,
                                      level_constraints[level],
                                      MyQuadrature(dof_h.get_fe().degree + 1),
                                      additional_data);

          mg_matrices[level].initialize(mg_mf_storage_level);
        }
    }
  else
    {
      const unsigned int nlevels = triangulation.n_global_levels();
      mg_matrices.resize(0, nlevels - 1);
      level_constraints.resize(0, nlevels - 1);

      const std::set<types::boundary_id> dirichlet_boundary = {0};
      mg_constrained_dofs.initialize(dof_handler);
      mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                         dirichlet_boundary);

      for (unsigned int level = 0; level < nlevels; ++level)
        {
          IndexSet relevant_dofs;
          DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                        level,
                                                        relevant_dofs);
          level_constraints[level].reinit(relevant_dofs);
          level_constraints[level].add_lines(
            mg_constrained_dofs.get_boundary_indices(level));
          level_constraints[level].close();

          typename MatrixFree<dim, LevelNumber>::AdditionalData additional_data;
          additional_data.tasks_parallel_scheme =
            MatrixFree<dim, LevelNumber>::AdditionalData::none;
          additional_data.mapping_update_flags =
            (update_gradients | update_JxW_values | update_quadrature_points);
          additional_data.mg_level = level;
          auto mg_mf_storage_level =
            std::make_shared<MatrixFree<dim, LevelNumber>>();
          mg_mf_storage_level->reinit(mapping,
                                      dof_handler,
                                      level_constraints[level],
                                      MyQuadrature(fe.degree + 1),
                                      additional_data);

          mg_matrices[level].initialize(mg_mf_storage_level,
                                        mg_constrained_dofs,
                                        level);
        }
    }
}



template <int dim>
void
LaplaceProblem<dim>::assemble_rhs()
{
  Timer time;

  system_rhs = 0;
  FEEvaluation<dim, -1> phi(*system_matrix.get_matrix_free());
  for (unsigned int cell = 0;
       cell < system_matrix.get_matrix_free()->n_cell_batches();
       ++cell)
    {
      phi.reinit(cell);
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(make_vectorized_array<double>(1.0), q);
      phi.integrate(EvaluationFlags::values);
      phi.distribute_local_to_global(system_rhs);
    }
  system_rhs.compress(VectorOperation::add);
}



template <int dim>
void
LaplaceProblem<dim>::setup_transfer()
{
  if (do_p_multigrid)
    {
      const unsigned int max_level = p_mg_level_degrees.size();
      mg_transfers_p.resize(0, max_level);
      for (unsigned int level = 0; level < max_level; ++level)
        mg_transfers_p[level + 1].reinit(level + 1 == max_level ?
                                           dof_handler :
                                           level_dof_handlers[level + 1],
                                         level_dof_handlers[level],
                                         level_constraints[level + 1],
                                         level_constraints[level]);
      mg_transfer_p =
        std::make_unique<MGTransferGlobalCoarsening<dim, VectorType>>(
          mg_transfers_p, [&](const unsigned int level, VectorType &vec) {
            mg_matrices[level].initialize_dof_vector(vec);
          });
    }
  else
    {
      mg_transfer.setup(mg_matrices, mg_constrained_dofs);
      std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
        partitioners(dof_handler.get_triangulation().n_global_levels());
      for (unsigned int level = 0; level < partitioners.size(); ++level)
        {
          VectorType vec;
          mg_matrices[level].initialize_dof_vector(vec);
          partitioners[level] = vec.get_partitioner();
        }
      mg_transfer.build(dof_handler, partitioners);
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
      if (level > 0)
        {
          smoother_data[level].smoothing_range     = 15.;
          smoother_data[level].degree              = 3;
          smoother_data[level].eig_cg_n_iterations = 12;
        }
      else if (!do_coarse_amg)
        {
          smoother_data[0].smoothing_range     = 1e-3;
          smoother_data[0].degree              = numbers::invalid_unsigned_int;
          smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
        }
      mg_matrices[level].compute_diagonal();
      smoother_data[level].preconditioner =
        mg_matrices[level].get_matrix_diagonal_inverse();
    }
  mg_smoother.initialize(mg_matrices, smoother_data);
}



template <int dim>
void
LaplaceProblem<dim>::setup_coarse_solver()
{
  if (do_coarse_amg)
    {
      mg_coarse = std::make_unique<
        MGCoarseSolverAMG<typename LevelMatrixType::value_type>>(
        mg_matrices[0].get_system_matrix(), 2);
    }
  else
    {
      auto coarse_solver =
        std::make_unique<MGCoarseGridApplySmoother<VectorType>>();
      coarse_solver->initialize(mg_smoother);
      mg_coarse = std::move(coarse_solver);
    }
}


template <int dim>
void
LaplaceProblem<dim>::solve()
{
  mg::Matrix<VectorType> mg_matrix(mg_matrices);

  SolverControl solver_control(500, 1e-8 * system_rhs.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
  constraints.set_zero(solution);

  if (do_p_multigrid)
    {
      Multigrid<VectorType> mg(
        mg_matrix, *mg_coarse, *mg_transfer_p, mg_smoother, mg_smoother);
      PreconditionMG<dim,
                     VectorType,
                     MGTransferGlobalCoarsening<dim, VectorType>>
        preconditioner(dof_handler, mg, *mg_transfer_p);

      cg.solve(system_matrix, solution, system_rhs, preconditioner);
    }
  else
    {
      Multigrid<VectorType> mg(
        mg_matrix, *mg_coarse, mg_transfer, mg_smoother, mg_smoother);

      MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
        mg_interface_matrices;
      mg_interface_matrices.resize(0, mg_matrices.max_level());
      for (unsigned int level = 0; level <= mg_matrices.max_level(); ++level)
        mg_interface_matrices[level].initialize(mg_matrices[level]);
      mg::Matrix<VectorType> mg_interface(mg_interface_matrices);
      mg.set_edge_matrices(mg_interface, mg_interface);

      PreconditionMG<dim, VectorType, MGTransferMF<dim, LevelMatrixType>>
        preconditioner(dof_handler, mg, mg_transfer);

      cg.solve(system_matrix, solution, system_rhs, preconditioner);
    }
  pcout << " Number of CG iterations: " << solver_control.last_step()
        << std::endl;
}



template <int dim>
void
LaplaceProblem<dim>::run(const unsigned int n_refinements,
                         const double       epsy,
                         const double       epsz)
{
  pcout << " Running Poisson problem in " << dim << "D with " << n_refinements
        << " refinements and degree " << fe.degree << std::endl;
  pcout << " Discretization parameters: degree=" << fe.degree
        << " epsy=" << epsy << " epsz=" << epsz << std::endl;
  const unsigned int n_vect_doubles = VectorizedArray<double>::size();
  const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

  pcout << " " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
        << " processes, vectorization over " << n_vect_doubles
        << " doubles = " << n_vect_bits << " bits ("
        << Utilities::System::get_current_vectorization_level() << ')'
        << std::endl;
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

  const unsigned int n_repeat = 50;
  timer["matvec_double"].start();
  for (unsigned int t = 0; t < n_repeat; ++t)
    system_matrix.vmult(system_rhs, solution);
  timer["matvec_double"].stop();

  using LevelNumber = typename LevelMatrixType::value_type;
  LinearAlgebra::distributed::Vector<LevelNumber> vec1, vec2;
  mg_matrices[mg_matrices.max_level()].initialize_dof_vector(vec1);
  vec2.reinit(vec1);
  timer["matvec_float"].start();
  for (unsigned int t = 0; t < n_repeat; ++t)
    mg_matrices[mg_matrices.max_level()].vmult(vec2, vec1);
  timer["matvec_float"].stop();

  mg_matrices[mg_matrices.max_level() - 1].initialize_dof_vector(vec1);
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
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");
      Vector<double> aspect_ratios =
        GridTools::compute_aspect_ratio_of_cells(mapping,
                                                 triangulation,
                                                 QGauss<dim>(3));
      data_out.add_data_vector(aspect_ratios, "aspect_ratio");

      data_out.build_patches(mapping);

      DataOutBase::VtkFlags flags;
      flags.compression_level = DataOutBase::CompressionLevel::best_speed;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);
      data_out.write_vtu_with_pvtu_record(
        "./", "solution", n_refinements, MPI_COMM_WORLD, 1);
    }
  timer["output"].stop();

  pcout << std::endl;
  for (const auto &entry : timer)
    {
      const auto min_max_avg =
        Utilities::MPI::min_max_avg(entry.second.wall_time(), MPI_COMM_WORLD);
      const unsigned int n_digits_ranks = Utilities::needed_digits(
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1);
      pcout << " Time " << std::left << std::setw(21) << entry.first
            << std::setw(11) << min_max_avg.min << " [p"
            << std::setw(n_digits_ranks) << min_max_avg.min_index << "] "
            << std::setw(11) << min_max_avg.avg << std::setw(11)
            << min_max_avg.max << " [p" << std::setw(n_digits_ranks)
            << min_max_avg.max_index << "]" << std::endl;
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
