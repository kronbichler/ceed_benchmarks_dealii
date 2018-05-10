
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/vector_tools.h>

#include "../common_code/curved_manifold.h"
#include "../common_code/poisson_operator.h"
#include "../common_code/diagonal_matrix_blocked.h"

using namespace dealii;


template <int dim, int fe_degree, int n_q_points>
void test(const unsigned int s,
          const bool short_output)
{
  Timer time;
  const unsigned int n_refine = s/3;
  const unsigned int remainder = s%3;
  Point<dim> p2;
  for (unsigned int d=0; d<remainder; ++d)
    p2[d] = 2;
  for (unsigned int d=remainder; d<dim; ++d)
    p2[d] = 1;

  MyManifold<dim> manifold;
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  std::vector<unsigned int> subdivisions(dim, 1);
  for (unsigned int d=0; d<remainder; ++d)
    subdivisions[d] = 2;
  GridGenerator::subdivided_hyper_rectangle(tria, subdivisions, Point<dim>(), p2);
  GridTools::transform(std_cxx11::bind(&MyManifold<dim>::push_forward, manifold,
                                       std_cxx11::_1),
                       tria);
  tria.set_all_manifold_ids(1);
  tria.set_manifold(1, manifold);
  tria.refine_global(n_refine);

  FE_Q<dim> fe_q(fe_degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_q);

  ConstraintMatrix constraints;
  IndexSet relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(dof_handler, 0, ZeroFunction<dim>(),
                                           constraints);
  constraints.close();
  std_cxx11::shared_ptr<MatrixFree<dim,double> > matrix_free(new MatrixFree<dim,double>());
  matrix_free->reinit(dof_handler, constraints, QGauss<1>(n_q_points),
                      typename MatrixFree<dim,double>::AdditionalData());

  Poisson::LaplaceOperator<dim,fe_degree,n_q_points,dim,double,
                           LinearAlgebra::distributed::BlockVector<double> > laplace_operator;
  laplace_operator.initialize(matrix_free);

  LinearAlgebra::distributed::BlockVector<double> input, output, output_test;
  input.reinit(dim);
  laplace_operator.initialize_dof_vector(input);
  output.reinit(dim);
  laplace_operator.initialize_dof_vector(output);
  output_test.reinit(dim);
  laplace_operator.initialize_dof_vector(output_test);
  for (unsigned int d=0; d<dim; ++d)
    for (unsigned int i=0; i<input.block(0).local_size(); ++i)
      if (!constraints.is_constrained(input.block(0).get_partitioner()->local_to_global(i)))
        input.block(d).local_element(i) = (i+d)%8;

  DiagonalMatrixBlocked<dim,double> diag_mat;
  diag_mat.diagonal = laplace_operator.compute_inverse_diagonal();

  Utilities::MPI::MinMaxAvg data =
    Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output == false)
    std::cout << "Setup time:         "
              << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")" << "s"
              << std::endl;

  ReductionControl solver_control(200, 1e-15, 1e-6);
  SolverCG<LinearAlgebra::distributed::BlockVector<double> > solver(solver_control);

  time.restart();
  try
    {
      solver.solve(laplace_operator, output, input, diag_mat);
    }
  catch (SolverControl::NoConvergence &e)
    {
      // prevent the solver to throw an exception in case we should need more
      // than 500 iterations
    }
  data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output==false)
    {
      std::cout << "Solve time:         "
                << data.min << " (p" << data.min_index << ") " << data.avg
                << " " << data.max << " (p" << data.max_index << ")" << "s"
                << std::endl;
      std::cout << "Time per iteration: "
                << data.min/solver_control.last_step()
                << " (p" << data.min_index << ") "
                << data.avg/solver_control.last_step()
                << " " << data.max/solver_control.last_step()
                << " (p" << data.max_index << ")" << "s"
                << std::endl;
    }
  const double solver_time = data.max;

  time.restart();
  for (unsigned int i=0; i<100; ++i)
    laplace_operator.vmult(input, output);
  const double t1 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max/100;

#ifdef SHOW_VARIANTS
  time.restart();
  for (unsigned int i=0; i<100; ++i)
    laplace_operator.vmult_basic(output_test, output);
  const double t2 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max/100;
  output_test -= input;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output==false)
    std::cout << "Error merged coefficient tensor:           "
              << output_test.linfty_norm() << std::endl;

  time.restart();
  for (unsigned int i=0; i<100; ++i)
    laplace_operator.vmult_construct_q(output_test, output);
  const double t3 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max/100;
  output_test -= input;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output==false)
    std::cout << "Error collocation evaluation of Jacobian:  "
              << output_test.linfty_norm() << std::endl;

  time.restart();
  for (unsigned int i=0; i<100; ++i)
    laplace_operator.vmult_linear_geo(output_test, output);
  const double t4 = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD).max/100;
  output_test -= input;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output==false)
    std::cout << "Error trilinear interpolation of Jacobian: "
              << output_test.linfty_norm() << std::endl;
#endif

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output == true)
    std::cout << std::setw(2) << fe_degree << " | " << std::setw(2) << n_q_points
              << " |" << std::setw(10) << tria.n_global_active_cells()
              << " |" << std::setw(11) << dim*dof_handler.n_dofs()
              << " | " << std::setw(11) << solver_time/solver_control.last_step()
              << " | " << std::setw(11) << dim*dof_handler.n_dofs()/solver_time*solver_control.last_step()
              << " | " << std::setw(4) << solver_control.last_step()
              << " | " << std::setw(11) << t1
#ifdef SHOW_VARIANTS
              << " | " << std::setw(11) << t2
              << " | " << std::setw(11) << t3
              << " | " << std::setw(11) << t4
#endif
              << std::endl;
}


template <int dim, int fe_degree, int n_q_points>
void do_test()
{
  unsigned int s =
    std::max(3U, static_cast<unsigned int>
             (1.000001*std::log(1024/fe_degree/fe_degree/fe_degree)/std::log(2)));
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef SHOW_VARIANTS
    std::cout << " p |  q | n_element |     n_dofs |     time/it |   dofs/s/it | itCG | time/matvec | timeMVmerge | timeMVcompu | timeMVlinear"
              << std::endl;
#else
    std::cout << " p |  q | n_element |     n_dofs |     time/it |   dofs/s/it | itCG | time/matvec"
#endif
              << std::endl;
  while (Utilities::fixed_power<dim>(fe_degree+1)*(1UL<<s)*dim
         < 6000000ULL*Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    {
      test<dim,fe_degree,n_q_points>(s, true);
      ++s;
    }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << std::endl;
}


int main(int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  do_test<3,1,3>();
  do_test<3,2,4>();
  do_test<3,3,5>();
  do_test<3,4,6>();
  do_test<3,5,7>();
  do_test<3,6,8>();
  do_test<3,7,9>();
  do_test<3,8,10>();

  return 0;
}
