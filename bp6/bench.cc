
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_renumbering.h>

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

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

#include "../common_code/curved_manifold.h"
#include "../common_code/diagonal_matrix_blocked.h"
#include "../common_code/poisson_operator.h"
#include "../common_code/solver_cg_optimized.h"

using namespace dealii;

// #define USE_SHMEM

template <int dim>
void
test(const unsigned int fe_degree,
     const unsigned int s,
     const bool         short_output,
     const MPI_Comm    &comm_shmem)
{
#ifndef USE_SHMEM
  (void)comm_shmem;
#endif

  warmup_code();
  const unsigned int n_q_points = fe_degree + 1;

  if (short_output == true)
    deallog.depth_console(0);
  else if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog.depth_console(2);

  Timer              time;
  const unsigned int n_refine  = s / 3;
  const unsigned int remainder = s % 3;
  Point<dim>         p2;
  for (unsigned int d = 0; d < remainder; ++d)
    p2[d] = 2;
  for (unsigned int d = remainder; d < dim; ++d)
    p2[d] = 1;

  MyManifold<dim>                           manifold;
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  std::vector<unsigned int>                 subdivisions(dim, 1);
  for (unsigned int d = 0; d < remainder; ++d)
    subdivisions[d] = 2;
  GridGenerator::subdivided_hyper_rectangle(tria, subdivisions, Point<dim>(), p2);
  GridTools::transform(std::bind(&MyManifold<dim>::push_forward, manifold, std::placeholders::_1),
                       tria);
  tria.set_all_manifold_ids(1);
  tria.set_manifold(1, manifold);
  tria.refine_global(n_refine);

  FE_Q<dim>            fe_q(fe_degree);
  MappingQGeneric<dim> mapping(1); // tri-linear mapping
  DoFHandler<dim>      dof_handler(tria);
  dof_handler.distribute_dofs(fe_q);

  AffineConstraints<double> constraints;
  IndexSet                  relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();
  typename MatrixFree<dim, double>::AdditionalData mf_data;

#ifdef USE_SHMEM
  mf_data.communicator_sm                = comm_shmem;
  mf_data.use_vector_data_exchanger_full = true;
#endif

  // renumber Dofs to minimize the number of partitions in import indices of
  // partitioner
  DoFRenumbering::matrix_free_data_locality(dof_handler, constraints, mf_data);

  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.clear();
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  std::shared_ptr<MatrixFree<dim, double>> matrix_free(new MatrixFree<dim, double>());
  matrix_free->reinit(mapping,
                      dof_handler,
                      constraints,
                      QGaussLobatto<1>(n_q_points),
                      typename MatrixFree<dim, double>::AdditionalData());

  Poisson::LaplaceOperator<dim, dim, double, LinearAlgebra::distributed::BlockVector<double>>
    laplace_operator;
  laplace_operator.initialize(matrix_free);

  LinearAlgebra::distributed::BlockVector<double> input, output, tmp;
  input.reinit(dim);
  laplace_operator.initialize_dof_vector(input);
  output.reinit(dim);
  laplace_operator.initialize_dof_vector(output);
  tmp.reinit(dim);
  laplace_operator.initialize_dof_vector(tmp);
  for (unsigned int d = 0; d < dim; ++d)
    for (unsigned int i = 0; i < input.block(0).locally_owned_size(); ++i)
      if (!constraints.is_constrained(input.block(0).get_partitioner()->local_to_global(i)))
        input.block(d).local_element(i) = (i + d) % 8;

  DiagonalMatrixBlocked<dim, double> diag_mat;
  diag_mat.diagonal = laplace_operator.compute_inverse_diagonal();
  if (short_output == false)
    {
      const double diag_norm = diag_mat.diagonal.l2_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Norm of diagonal for preconditioner: " << diag_norm << std::endl;
    }

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == false)
    std::cout << "Setup time:         " << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")"
              << "s" << std::endl;

  ReductionControl                                          solver_control(100, 1e-15, 1e-8);
  SolverCG<LinearAlgebra::distributed::BlockVector<double>> solver(solver_control);

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

  /*
  SolverCGOptimized<LinearAlgebra::distributed::BlockVector<double>> solver2(solver_control);
  double                                                             solver_time2 = 1e10;
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
  */

  SolverCGFullMerge<LinearAlgebra::distributed::BlockVector<double>> solver4(solver_control);
  double                                                             solver_time4 = 1e10;
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
  LIKWID_MARKER_START("matvec_quad");
#  endif
  time.restart();
  for (unsigned int i = 0; i < 100; ++i)
    laplace_operator.vmult_quadratic(tmp, output);
  const double t5 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;
#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("matvec_quad");
#  endif

  tmp -= input;
  if (short_output == false)
    {
      const double norm = tmp.linfty_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Error quadratic evaluation of Jacobian:    " << norm << std::endl;
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

  tmp -= input;
  if (short_output == false)
    {
      const double norm = tmp.linfty_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Error trilinear interpolation of Jacobian: " << norm << std::endl;
    }
#endif

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == true)
    std::cout << std::setw(2) << fe_degree << " | " << std::setw(2) << n_q_points   //
              << " |" << std::setw(10) << tria.n_global_active_cells()              //
              << " |" << std::setw(11) << dim * dof_handler.n_dofs()                //
              << " | " << std::setw(11) << solver_time / solver_control.last_step() //
              << " | " << std::setw(11)
              << dim * dof_handler.n_dofs() / solver_time4 * solver_control.last_step() //
              << " | " << std::setw(11) << solver_time4 / solver_control.last_step()    //
              << " | " << std::setw(4) << solver_control.last_step()                    //
              << " | " << std::setw(11) << matvec_time                                  //
#ifdef SHOW_VARIANTS
              << " | " << std::setw(11) << t2 //
              << " | " << std::setw(11) << t3 //
              << " | " << std::setw(11) << t4 //
              << " | " << std::setw(11) << t5 //
#endif
              << std::endl;
}


template <int dim>
void
do_test(const unsigned int fe_degree, const int s_in, const bool compact_output)
{
  MPI_Comm comm_shmem = MPI_COMM_SELF;

#ifdef USE_SHMEM
  MPI_Comm_split_type(MPI_COMM_WORLD,
                      MPI_COMM_TYPE_SHARED,
                      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                      MPI_INFO_NULL,
                      &comm_shmem);
#endif

  if (s_in < 1)
    {
      unsigned int s =
        1 + static_cast<unsigned int>(std::log2(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)));
      // std::max(3U, static_cast<unsigned int>
      //         (std::log2(1024/fe_degree/fe_degree/fe_degree)));
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef SHOW_VARIANTS
        std::cout
          << " p |  q | n_element |     n_dofs |     time/it |   dofs/s/it | opt_time/it | itCG | time/matvec | timeMVbasic | timeMVcompu | timeMVmerge | timeMVquad"
          << std::endl;
#else
        std::cout
          << " p |  q | n_element |     n_dofs |     time/it |   dofs/s/it | opt_time/it | itCG | time/matvec"
#endif
      << std::endl;
      while (Utilities::fixed_power<dim>(fe_degree + 1) * (1UL << s) * dim <
             6000000ULL * Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        {
          test<dim>(fe_degree, s, compact_output, comm_shmem);
          ++s;
        }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << std::endl << std::endl;
    }
  else
    test<dim>(fe_degree, s_in, compact_output, comm_shmem);

#ifdef USE_SHMEM
  MPI_Comm_free(&comm_shmem);
#endif
}


int
main(int argc, char **argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
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

  do_test<3>(degree, s, compact_output);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
