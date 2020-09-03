
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/vector_tools.h>

//#define USE_STD_SIMD

#ifdef USE_STD_SIMD
#  include <experimental/simd>
#endif

#include "../common_code/curved_manifold.h"
#include "../common_code/poisson_operator.h"
#include "../common_code/renumber_dofs_for_mf.h"
#include "../common_code/solver_cg_optimized.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

using namespace dealii;

#include "../common_code/create_triangulation.h"

// VERSION:
//   0: p:d:t + vectorized over elements;
//   1: p:f:t + vectorized within element (not optimized)
#define VERSION 0

#if VERSION == 0

#  ifdef USE_STD_SIMD
typedef std::experimental::native_simd<double> VectorizedArrayType;
#  else
typedef dealii::VectorizedArray<double> VectorizedArrayType;
#  endif

#elif VERSION == 1
typedef dealii::VectorizedArray<double, 1> VectorizedArrayType;
#endif



template <int dim, int fe_degree, int n_q_points>
void
test(const unsigned int s, const bool short_output)
{
  warmup_code();

  if (short_output == true)
    deallog.depth_console(0);
  else if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog.depth_console(2);

  Timer           time;
  MyManifold<dim> manifold;

  const auto tria = create_triangulation(s, manifold, VERSION);

  FE_Q<dim>       fe_q(fe_degree);
  DoFHandler<dim> dof_handler(*tria);
  dof_handler.distribute_dofs(fe_q);

  AffineConstraints<double> constraints;
  IndexSet                  relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(dof_handler, 0, ZeroFunction<dim>(), constraints);
  constraints.close();
  typename MatrixFree<dim, double, VectorizedArrayType>::AdditionalData mf_data;

  // renumber Dofs to minimize the number of partitions in import indices of
  // partitioner
  Renumber<dim, double> renum(0, 1, 2);
  renum.renumber(dof_handler, constraints, mf_data);

  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.clear();
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(dof_handler, 0, ZeroFunction<dim>(), constraints);
  constraints.close();

  std::shared_ptr<MatrixFree<dim, double, VectorizedArrayType>> matrix_free(
    new MatrixFree<dim, double, VectorizedArrayType>());
  matrix_free->reinit(dof_handler, constraints, QGaussLobatto<1>(n_q_points), mf_data);

  Poisson::LaplaceOperator<dim,
                           fe_degree,
                           n_q_points,
                           1,
                           double,
                           LinearAlgebra::distributed::Vector<double>,
                           VectorizedArrayType>
    laplace_operator;
  laplace_operator.initialize(matrix_free, constraints);

  LinearAlgebra::distributed::Vector<double> input, output, tmp;
  laplace_operator.initialize_dof_vector(input);
  laplace_operator.initialize_dof_vector(output);
  laplace_operator.initialize_dof_vector(tmp);
  for (unsigned int i = 0; i < input.local_size(); ++i)
    if (!constraints.is_constrained(input.get_partitioner()->local_to_global(i)))
      input.local_element(i) = (i) % 8;

  DiagonalMatrix<LinearAlgebra::distributed::Vector<double>> diag_mat;
  diag_mat.get_vector() = laplace_operator.compute_inverse_diagonal();
  if (short_output == false)
    {
      const double diag_norm = diag_mat.get_vector().l2_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Norm of diagonal for preconditioner: " << diag_norm << std::endl;
    }

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == false)
    std::cout << "Setup time:         " << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")"
              << "s" << std::endl;

  ReductionControl                                     solver_control(100, 1e-15, 1e-8);
  SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("cg_solver");
#endif
  double solver_time = 1e10;
  for (unsigned int t = 0; t < 2; ++t)
    {
      output = 0;
      time.restart();
      try
        {
          solver.solve(laplace_operator, output, input, diag_mat);
        }
      catch (SolverControl::NoConvergence &e)
        {
          // prevent the solver to throw an exception in case we should need more
          // than 100 iterations
        }
      data        = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      solver_time = std::min(data.max, solver_time);
    }
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("cg_solver");
#endif
  if (short_output == false)
    {
      tmp.equ(-1.0, input);
      laplace_operator.vmult_add(tmp, output);
      deallog << "True residual norm: " << tmp.l2_norm() << std::endl;
    }
  const unsigned int iterations_basic = solver_control.last_step();

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == false)
    {
      std::cout << "Solve time:         " << data.min << " (p" << data.min_index << ") " << data.avg
                << " " << data.max << " (p" << data.max_index << ")"
                << "s" << std::endl;
      std::cout << "Time per iteration: " << data.min / solver_control.last_step() << " (p"
                << data.min_index << ") " << data.avg / solver_control.last_step() << " "
                << data.max / solver_control.last_step() << " (p" << data.max_index << ")"
                << "s" << std::endl;
    }

  SolverCGOptimized<LinearAlgebra::distributed::Vector<double>> solver2(solver_control);
  double                                                        solver_time2 = 1e10;
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("cg_solver_opt");
#endif
  for (unsigned int t = 0; t < 2; ++t)
    {
      output = 0;
      time.restart();
      try
        {
          solver2.solve(laplace_operator, output, input, diag_mat);
        }
      catch (SolverControl::NoConvergence &e)
        {
          // prevent the solver to throw an exception in case we should need more
          // than 100 iterations
        }
      data         = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      solver_time2 = std::min(data.max, solver_time2);
    }
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("cg_solver_opt");
#endif
  if (short_output == false)
    {
      tmp.equ(-1.0, input);
      laplace_operator.vmult_add(tmp, output);
      deallog << "True residual norm: " << tmp.l2_norm() << std::endl;
    }
  AssertThrow(std::abs((int)solver_control.last_step() - (int)iterations_basic) < 2,
              ExcMessage("Iteration numbers differ " + std::to_string(solver_control.last_step()) +
                         " vs default solver " + std::to_string(iterations_basic)));

  SolverCGFullMerge<LinearAlgebra::distributed::Vector<double>> solver4(solver_control);
  double                                                        solver_time4 = 1e10;
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("cg_solver_optm");
#endif
  for (unsigned int t = 0; t < 4; ++t)
    {
      output = 0;
      time.restart();
      try
        {
          solver4.solve(laplace_operator, output, input, diag_mat);
        }
      catch (SolverControl::NoConvergence &e)
        {
          // prevent the solver to throw an exception in case we should need more
          // than 100 iterations
        }
      data         = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      solver_time4 = std::min(data.max, solver_time4);
    }
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("cg_solver_optm");
#endif
  if (short_output == false)
    {
      tmp.equ(-1.0, input);
      laplace_operator.vmult_add(tmp, output);
      deallog << "True residual norm: " << tmp.l2_norm() << std::endl;
    }
  AssertThrow(std::abs((int)solver_control.last_step() - (int)iterations_basic) < 2,
              ExcMessage("Iteration numbers differ " + std::to_string(solver_control.last_step()) +
                         " vs default solver " + std::to_string(iterations_basic)));

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("matvec");
#endif
  double matvec_time = 1e10;
  for (unsigned int t = 0; t < 2; ++t)
    {
      time.restart();
      for (unsigned int i = 0; i < 50; ++i)
        laplace_operator.vmult(input, output);
      data        = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      matvec_time = std::min(data.max / 50, matvec_time);
    }
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("matvec");
#endif

#ifdef SHOW_VARIANTS
#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("matvec_basic");
#  endif
  time.restart();
  for (unsigned int i = 0; i < 100; ++i)
    laplace_operator.vmult_basic(tmp, output);
#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("matvec_basic");
#  endif
  const double t2 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;
  tmp -= input;
  if (short_output == false)
    {
      const double norm = tmp.linfty_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Error merged coefficient tensor:           " << norm << std::endl;
    }

#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("matvec_q");
#  endif
  time.restart();
  for (unsigned int i = 0; i < 100; ++i)
    laplace_operator.vmult_construct_q(tmp, output);
  const double t3 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;
#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("matvec_q");
#  endif

  tmp -= input;
  if (short_output == false)
    {
      const double norm = tmp.linfty_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Error collocation evaluation of Jacobian:  " << norm << std::endl;
    }

#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("matvec_merge");
#  endif
  time.restart();
  for (unsigned int i = 0; i < 100; ++i)
    laplace_operator.vmult_merged(tmp, output);
  const double t4 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;

#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("matvec_merge");
#  endif

#endif

  tmp -= input;
  if (short_output == false)
    {
      const double norm = tmp.linfty_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Error trilinear interpolation of Jacobian: " << norm << std::endl;
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == true)
    std::cout << std::setw(2) << fe_degree << " | " << std::setw(2) << n_q_points   //
              << " |" << std::setw(10) << tria->n_global_active_cells()             //
              << " |" << std::setw(11) << dof_handler.n_dofs()                      //
              << " | " << std::setw(11) << solver_time / solver_control.last_step() //
              << " | " << std::setw(11)
              << dof_handler.n_dofs() / solver_time2 * solver_control.last_step()    //
              << " | " << std::setw(11) << solver_time2 / solver_control.last_step() //
              << " | " << std::setw(11) << solver_time4 / solver_control.last_step() //
              << " | " << std::setw(4) << solver_control.last_step()                 //
              << " | " << std::setw(11) << matvec_time                               //
#ifdef SHOW_VARIANTS
              << " | " << std::setw(11) << t2 //
              << " | " << std::setw(11) << t3 //
              << " | " << std::setw(11) << t4 //
#endif
              << std::endl;
}


template <int dim, int fe_degree, int n_q_points>
void
do_test(const int s_in, const bool compact_output)
{
  if (s_in < 1)
    {
      unsigned int s =
        std::max(3U,
                 static_cast<unsigned int>(std::log2(1024 / fe_degree / fe_degree / fe_degree)));
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef SHOW_VARIANTS
        std::cout
          << " p |  q | n_element |     n_dofs |     time/it |   dofs/s/it | opt_time/it | opm_time/it | itCG | time/matvec | timeMVbasic | timeMVcompu | timeMVmerged"
#else
        std::cout
          << " p |  q | n_element |     n_dofs |     time/it |   dofs/s/it | opt_time/it | opm_time/it | itCG | time/matvec"
#endif
          << std::endl;
      while ((2 + Utilities::fixed_power<dim>(fe_degree + 1)) * (1UL << s) <
             6000000ULL * Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        {
          test<dim, fe_degree, n_q_points>(s, compact_output);
          ++s;
        }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << std::endl << std::endl;
    }
  else
    test<dim, fe_degree, n_q_points>(s_in, compact_output);
}


int
main(int argc, char **argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#endif

  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  unsigned int degree         = 1;
  unsigned int s              = -1;
  bool         compact_output = true;
  if (argc > 1)
    degree = std::atoi(argv[1]);
  if (argc > 2)
    s = std::atoi(argv[2]);
  if (argc > 3)
    compact_output = std::atoi(argv[3]);

  if (degree == 1)
    do_test<3, 1, 2>(s, compact_output);
  else if (degree == 2)
    do_test<3, 2, 3>(s, compact_output);
  else if (degree == 3)
    do_test<3, 3, 4>(s, compact_output);
  else if (degree == 4)
    do_test<3, 4, 5>(s, compact_output);
  else if (degree == 5)
    do_test<3, 5, 6>(s, compact_output);
  else if (degree == 6)
    do_test<3, 6, 7>(s, compact_output);
  else if (degree == 7)
    do_test<3, 7, 8>(s, compact_output);
  else if (degree == 8)
    do_test<3, 8, 9>(s, compact_output);
  else if (degree == 9)
    do_test<3, 9, 10>(s, compact_output);
  else if (degree == 10)
    do_test<3, 10, 11>(s, compact_output);
  else if (degree == 11)
    do_test<3, 11, 12>(s, compact_output);
  else if (degree == 12)
    do_test<3, 12, 13>(s, compact_output);
  else if (degree == 13)
    do_test<3, 13, 14>(s, compact_output);
  else if (degree == 14)
    do_test<3, 14, 15>(s, compact_output);
  else if (degree == 15)
    do_test<3, 15, 16>(s, compact_output);
  else
    AssertThrow(false, ExcMessage("Only degrees up to 11 implemented"));

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
