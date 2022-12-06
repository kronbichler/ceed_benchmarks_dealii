
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

//#define USE_STD_SIMD

#ifdef USE_STD_SIMD
#  include <experimental/simd>
#endif

#include "../common_code/curved_manifold.h"
#include "../common_code/poisson_operator.h"
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

using Number = float;

#if VERSION == 0

#  ifdef USE_STD_SIMD
typedef std::experimental::native_simd<Number> VectorizedArrayType;
#  else
typedef dealii::VectorizedArray<Number> VectorizedArrayType;
#  endif

#elif VERSION == 1
typedef dealii::VectorizedArray<Number, 1> VectorizedArrayType;
#endif

#define USE_SHMEM
#define SHOW_VARIANTS

template <int dim, int fe_degree, int n_q_points>
void
test(const unsigned int s, const bool short_output, const MPI_Comm &comm_shmem)
{
#ifndef USE_SHMEM
  (void)comm_shmem;
#endif

  warmup_code();

  if (short_output == true)
    deallog.depth_console(0);
  else if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog.depth_console(2);

  Timer           time;
  MyManifold<dim> manifold;

  const auto tria = create_triangulation(s, manifold, VERSION);

  FE_Q<dim>            fe_q(fe_degree);
  MappingQGeneric<dim> mapping(1); // tri-linear mapping
  DoFHandler<dim>      dof_handler(*tria);
  dof_handler.distribute_dofs(fe_q);

  AffineConstraints<double> constraints;
  constraints.close();
  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData mf_data;

  DoFRenumbering::matrix_free_data_locality(dof_handler, constraints, mf_data);

#ifdef USE_SHMEM
  mf_data.communicator_sm = comm_shmem;
  // mf_data.use_vector_data_exchanger_full = true;
#endif

  std::shared_ptr<MatrixFree<dim, Number, VectorizedArrayType>> matrix_free(
    new MatrixFree<dim, Number, VectorizedArrayType>());
  matrix_free->reinit(mapping, dof_handler, constraints, QGaussLobatto<1>(n_q_points), mf_data);

  Poisson::LaplaceOperator<dim,
                           fe_degree,
                           n_q_points,
                           1,
                           Number,
                           LinearAlgebra::distributed::Vector<Number>,
                           VectorizedArrayType>
    laplace_operator;
  laplace_operator.initialize(matrix_free, constraints);

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == false)
    std::cout << "Setup time:         " << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")"
              << "s" << std::endl;

  AlignedVector<VectorizedArray<Number>> x(matrix_free->n_cell_batches() * fe_q.dofs_per_cell),
    Ax(matrix_free->n_cell_batches() * fe_q.dofs_per_cell);

  MPI_Barrier(MPI_COMM_WORLD);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("matvec");
#endif
  double matvec_time = 1e10;
  for (unsigned int t = 0; t < 2; ++t)
    {
      time.restart();
      for (unsigned int i = 0; i < 50; ++i)
        laplace_operator.apply_local(Ax, x);
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
    laplace_operator.apply_basic(Ax, x);
#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("matvec_basic");
#  endif
  const double t2 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;

#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("matvec_q");
#  endif
  time.restart();
  for (unsigned int i = 0; i < 100; ++i)
    laplace_operator.apply_q(Ax, x);
  const double t3 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;
#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("matvec_q");
#  endif

#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("matvec_q1");
#  endif
  time.restart();
  for (unsigned int i = 0; i < 100; ++i)
    laplace_operator.apply_q1(Ax, x);
  const double t4 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;

#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("matvec_q1");
#  endif

#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("matvec_q2");
#  endif
  time.restart();
  for (unsigned int i = 0; i < 100; ++i)
    laplace_operator.apply_q2(Ax, x);
  const double t5 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;

#  ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("matvec_q2");
#  endif

#endif

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == true)
    std::cout << std::setw(2) << fe_degree << "  " << std::setw(2) << n_q_points //
              << " " << std::setw(10) << tria->n_global_active_cells()           //
              << " " << std::setw(11)
              << (tria->n_global_active_cells() *     //
                  Utilities::pow(fe_q.degree, dim))   //
              << "  " << std::setw(11) << matvec_time //
#ifdef SHOW_VARIANTS
              << "  " << std::setw(11) << t2 //
              << "  " << std::setw(11) << t3 //
              << "  " << std::setw(11) << t4 //
              << "  " << std::setw(11) << t5 //
#endif
              << std::endl;
}


template <int dim, int fe_degree, int n_q_points>
void
do_test(const int s_in, const bool compact_output)
{
  MPI_Comm comm_shmem;

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
        std::max(3U,
                 static_cast<unsigned int>(std::log2(1024 / fe_degree / fe_degree / fe_degree)));
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef SHOW_VARIANTS
        std::cout
          << " p   q  n_element      n_dofs  matvec effc  matvec basi  matvec q     matvec q1    matvec q2"
#else
        std::cout << " p   q  n_element      n_dofs  matvec"
#endif
          << std::endl;
      while ((2 + Utilities::fixed_power<dim>(fe_degree + 1)) * (1UL << s) <
             6000000ULL * Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        {
          test<dim, fe_degree, n_q_points>(s, compact_output, comm_shmem);
          ++s;
        }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << std::endl << std::endl;
    }
  else
    test<dim, fe_degree, n_q_points>(s_in, compact_output, comm_shmem);

#ifdef USE_SHMEM
  MPI_Comm_free(&comm_shmem);
#endif
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
