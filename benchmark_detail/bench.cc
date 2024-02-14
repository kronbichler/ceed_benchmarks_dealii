
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
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

#define USE_SHMEM

template <typename VectorType>
class SolverCGDetail : public dealii::SolverBase<VectorType>
{
public:
  /**
   * Declare type for container size.
   */
  using size_type = dealii::types::global_dof_index;

  /**
   * Constructor.
   */
  SolverCGDetail(dealii::SolverControl &cn)
    : dealii::SolverBase<VectorType>(cn)
  {}


  /**
   * Virtual destructor.
   */
  virtual ~SolverCGDetail() override = default;

  /**
   * Solve the linear system $Ax=b$ for x.
   */
  template <typename MatrixType, typename PreconditionerType>
  std::vector<double>
  solve(const MatrixType         &A,
        VectorType               &x,
        const VectorType         &b,
        const PreconditionerType &preconditioner)
  {
    dealii::SolverControl::State conv = dealii::SolverControl::iterate;
    using number                      = typename VectorType::value_type;

    // Memory allocation
    typename dealii::VectorMemory<VectorType>::Pointer g_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer d_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer h_pointer(this->memory);

    // define some aliases for simpler access
    VectorType &g = *g_pointer;
    VectorType &d = *d_pointer;
    VectorType &h = *h_pointer;

    int    it  = 0;
    double res = -std::numeric_limits<double>::max();

    // resize the vectors, but do not set
    // the values since they'd be overwritten
    // soon anyway.
    g.reinit(x, true);
    d.reinit(x, true);
    h.reinit(x, true);

    std::vector<double> times(4);

    number gh, beta;

    // compute residual. if vector is zero, then short-circuit the full
    // computation
    if (!x.all_zero())
      {
        A.vmult(g, x);
        g.add(-1., b);
      }
    else
      {
        Timer time;
        g.equ(-1., b);
        times[1] += time.wall_time();
      }
    {
      Timer time;
      res = g.l2_norm();
      times[2] += time.wall_time();
    }

    conv = this->iteration_status(0, res, x);
    if (conv == dealii::SolverControl::iterate)
      {
        {
          Timer time;
          preconditioner.vmult(h, g);
          times[3] += time.wall_time();
        }
        {
          Timer time;
          d.equ(-1., h);
          times[1] += time.wall_time();
        }

        {
          Timer time;
          gh = g * h;
          times[2] += time.wall_time();
        }
      }

    while (conv == dealii::SolverControl::iterate)
      {
        it++;
        {
          Timer time;
          A.vmult(h, d);
          times[0] += time.wall_time();
        }
        number alpha = 0;
        {
          Timer time;
          alpha = h * d;
          times[2] += time.wall_time();
        }

        Assert(std::abs(alpha) != 0., dealii::ExcDivideByZero());
        alpha = gh / alpha;

        {
          Timer time;
          x.add(alpha, d);
          g.add(alpha, h);
          times[1] += time.wall_time();
        }
        {
          Timer time;
          res = g.l2_norm();
          times[2] += time.wall_time();
        }

        conv = this->iteration_status(it, res, x);

        if (conv != SolverControl::iterate)
          break;

        {
          Timer time;
          preconditioner.vmult(h, g);
          times[3] += time.wall_time();
        }

        beta = gh;
        Assert(std::abs(beta) != 0., ExcDivideByZero());
        {
          Timer time;
          gh = g * h;
          times[2] += time.wall_time();
        }
        beta = gh / beta;
        {
          Timer time;
          d.sadd(beta, -1., h);
          times[1] += time.wall_time();
        }
      }

    Utilities::MPI::sum(times, g.get_mpi_communicator(), times);
    return times;
  }
};


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
  const unsigned int n_q_points = fe_degree + 2;

  if (short_output == true)
    deallog.depth_console(0);
  else if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog.depth_console(2);

  Timer              time;
  unsigned int       n_refine  = s / 6;
  const unsigned int remainder = s % 6;

  MyManifold<dim>                           manifold;
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  std::vector<unsigned int>                 subdivisions(dim, 1);
  if (remainder == 1 && s > 1)
    {
      subdivisions[0] = 3;
      subdivisions[1] = 2;
      subdivisions[2] = 2;
      n_refine -= 1;
    }
  if (remainder == 2)
    subdivisions[0] = 2;
  else if (remainder == 3)
    subdivisions[0] = 3;
  else if (remainder == 4)
    subdivisions[0] = subdivisions[1] = 2;
  else if (remainder == 5)
    {
      subdivisions[0] = 3;
      subdivisions[1] = 2;
    }

  Point<dim> p2;
  for (unsigned int d = 0; d < dim; ++d)
    p2[d] = subdivisions[d];
  GridGenerator::subdivided_hyper_rectangle(tria, subdivisions, Point<dim>(), p2);
  GridTools::transform(std::bind(&MyManifold<dim>::push_forward, manifold, std::placeholders::_1),
                       tria);
  tria.set_all_manifold_ids(1);
  tria.set_manifold(1, manifold);
  tria.refine_global(n_refine);

  FE_Q<dim>            fe_scalar(fe_degree);
  FESystem<dim>        fe_q(fe_scalar, dim);
  MappingQGeneric<dim> mapping(1); // tri-linear mapping
  DoFHandler<dim>      dof_handler(tria);
  dof_handler.distribute_dofs(fe_q);

  AffineConstraints<double> constraints;
  IndexSet                  relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(dim), constraints);
  constraints.close();
  typename MatrixFree<dim, double>::AdditionalData mf_data;

#ifdef USE_SHMEM
  mf_data.communicator_sm = comm_shmem;
#endif

  // renumber Dofs to minimize the number of partitions in import indices of
  // partitioner
  DoFRenumbering::matrix_free_data_locality(dof_handler, constraints, mf_data);

  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.clear();
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(dim), constraints);
  constraints.close();

  std::shared_ptr<MatrixFree<dim, double>> matrix_free(new MatrixFree<dim, double>());

  // create preconditioner based on the diagonal of the GLL quadrature with
  // fe_degree+1 points
  DiagonalMatrixBlocked<dim, double> diag_mat;
  {
    matrix_free->reinit(
      mapping, dof_handler, constraints, QGaussLobatto<1>(fe_degree + 1), mf_data);

    Poisson::LaplaceOperator<dim, dim, double, LinearAlgebra::distributed::Vector<double>>
      laplace_operator;
    laplace_operator.initialize(matrix_free);

    const auto vector = laplace_operator.compute_inverse_diagonal();
    IndexSet   reduced(vector.size() / dim);
    reduced.add_range(vector.get_partitioner()->local_range().first / dim,
                      vector.get_partitioner()->local_range().second / dim);
    reduced.compress();
    diag_mat.diagonal.reinit(reduced, vector.get_mpi_communicator());
    for (unsigned int i = 0; i < reduced.n_elements(); ++i)
      diag_mat.diagonal.local_element(i) = vector.local_element(i * dim);
  }
  if (short_output == false)
    {
      const double diag_norm = diag_mat.diagonal.l2_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Norm of diagonal for preconditioner: " << diag_norm << std::endl;
    }

  // now go back to the actual operator with n_q_points points
  matrix_free->reinit(mapping, dof_handler, constraints, QGauss<1>(n_q_points), mf_data);

  Poisson::LaplaceOperator<dim, dim, double, LinearAlgebra::distributed::Vector<double>>
    laplace_operator;
  laplace_operator.initialize(matrix_free);

  LinearAlgebra::distributed::Vector<double> input, output, tmp;
  laplace_operator.initialize_dof_vector(input);
  laplace_operator.initialize_dof_vector(output);
  laplace_operator.initialize_dof_vector(tmp);
  for (unsigned int i = 0; i < input.locally_owned_size(); ++i)
    if (!constraints.is_constrained(input.get_partitioner()->local_to_global(i)))
      input.local_element(i) = i % 8;

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == false)
    std::cout << "Setup time:         " << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")"
              << "s" << std::endl;

  ReductionControl                                           solver_control(100, 1e-15, 1e-8);
  SolverCGDetail<LinearAlgebra::distributed::Vector<double>> solver(solver_control);

  double              solver_time = 1e10;
  std::vector<double> times;
  for (unsigned int t = 0; t < 4; ++t)
    {
      output = 0;
      time.restart();
      const auto my_times = solver.solve(laplace_operator, output, input, diag_mat);
      data                = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      if (data.max < solver_time)
        {
          times       = my_times;
          solver_time = data.max;
        }
    }

  double matvec_time = 1e10;
  for (unsigned int t = 0; t < 2; ++t)
    {
      time.restart();
      for (unsigned int i = 0; i < 50; ++i)
        laplace_operator.vmult(input, output);
      data        = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      matvec_time = std::min(data.max / 50, matvec_time);
    }

  time.restart();
  for (unsigned int i = 0; i < 100; ++i)
    laplace_operator.vmult_basic(tmp, output);
  const double t2 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;
  tmp -= input;
  if (short_output == false)
    {
      const double norm = tmp.linfty_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Error merged coefficient tensor:           " << norm << std::endl;
    }

  time.restart();
  for (unsigned int i = 0; i < 100; ++i)
    laplace_operator.vmult_construct_q(tmp, output);
  const double t3 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;
  tmp -= input;
  if (short_output == false)
    {
      const double norm = tmp.linfty_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Error collocation evaluation of Jacobian:  " << norm << std::endl;
    }

  time.restart();
  for (unsigned int i = 0; i < 100; ++i)
    laplace_operator.vmult_quadratic(tmp, output);
  const double t5 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;

  tmp -= input;
  if (short_output == false)
    {
      const double norm = tmp.linfty_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Error quadratic evaluation of Jacobian:    " << norm << std::endl;
    }

  time.restart();
  for (unsigned int i = 0; i < 100; ++i)
    laplace_operator.vmult_merged(tmp, output);
  const double t4 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;
  tmp -= input;
  if (short_output == false)
    {
      const double norm = tmp.linfty_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Error trilinear interpolation of Jacobian: " << norm << std::endl;
    }

  tria.clear();
  GridGenerator::subdivided_hyper_rectangle(tria, subdivisions, Point<dim>(), p2);
  tria.refine_global(n_refine);
  dof_handler.distribute_dofs(fe_q);
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(dim), constraints);
  constraints.close();
  DoFRenumbering::matrix_free_data_locality(dof_handler, constraints, mf_data);

  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.clear();
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(dim), constraints);
  constraints.close();
  // now go back to the actual operator with n_q_points points
  matrix_free->reinit(mapping, dof_handler, constraints, QGauss<1>(fe_degree + 1), mf_data);

  Poisson::LaplaceOperator<dim, dim, double, LinearAlgebra::distributed::Vector<double>>
    laplace_operator2;
  laplace_operator2.initialize(matrix_free);
  laplace_operator2.initialize_dof_vector(tmp);
  laplace_operator2.initialize_dof_vector(output);
  time.restart();
  for (unsigned int i = 0; i < 100; ++i)
    laplace_operator2.vmult_basic(tmp, output);
  const double t6 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max / 100;

  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const unsigned int nits    = solver_control.last_step();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == true)
    std::cout << std::setprecision(4) << std::setw(2) << fe_degree << " " //
              << " " << std::setw(9) << dof_handler.n_dofs()              //
              << " " << std::setw(9) << solver_time / nits                //
              << " " << std::setw(9) << times[0] / n_ranks / nits         //
              << " " << std::setw(9) << times[1] / n_ranks / nits         //
              << " " << std::setw(9) << times[2] / n_ranks / nits         //
              << " " << std::setw(9) << times[3] / n_ranks / nits         //
              << " " << std::setw(9) << matvec_time                       //
              << " " << std::setw(9) << t2                                //
              << " " << std::setw(9) << t3                                //
              << " " << std::setw(9) << t4                                //
              << " " << std::setw(9) << t5                                //
              << " " << std::setw(9) << t6                                //
              << std::endl;
}


template <int dim>
void
do_test(const unsigned int fe_degree, const int s_in, const bool compact_output)
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
        1 + static_cast<unsigned int>(std::log2(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)));
      // std::max(3U, static_cast<unsigned int>
      //         (std::log2(1024/fe_degree/fe_degree/fe_degree)));
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout
          << " p       dofs   timeSol   timeMVs  timeVecU   timeDot  timePrec  timeMVqu  timeMVba  timeMVco  timeMVme  timeMVli  timeMVca"
          << std::endl;
      while (Utilities::fixed_power<dim>(fe_degree + 1) * (1UL << (s / 2)) * dim <
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
