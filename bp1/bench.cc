
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "../common_code/curved_manifold.h"
#include "../common_code/mass_operator.h"

using namespace dealii;



template <int dim, int fe_degree, int n_q_points>
void test(const unsigned int s,
          const bool short_output)
{
  warmup_code();

  if (short_output == true)
    deallog.depth_console(0);
  else if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog.depth_console(2);

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
  GridTools::transform(std::bind(&MyManifold<dim>::push_forward, manifold,
                                 std::placeholders::_1),
                       tria);
  tria.set_all_manifold_ids(1);
  tria.set_manifold(1, manifold);
  tria.refine_global(n_refine);

  FE_Q<dim> fe(fe_degree);
  MappingQGeneric<dim> mapping(1);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  ConstraintMatrix constraints;
  constraints.close();
  std::shared_ptr<MatrixFree<dim,double> > matrix_free(new MatrixFree<dim,double>());
  matrix_free->reinit(mapping, dof_handler, constraints, QGauss<1>(n_q_points),
                      typename MatrixFree<dim,double>::AdditionalData());

  Mass::MassOperator<dim,fe_degree,n_q_points> mass_operator;
  mass_operator.initialize(matrix_free);

  LinearAlgebra::distributed::Vector<double> input, output;
  matrix_free->initialize_dof_vector(input);
  matrix_free->initialize_dof_vector(output);
  for (unsigned int i=0; i<input.local_size(); ++i)
    input.local_element(i) = i%8;

  Utilities::MPI::MinMaxAvg data =
    Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output == false)
    std::cout << "Setup time:         "
              << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")" << "s"
              << std::endl;

  ReductionControl solver_control(100, 1e-15, 1e-8);
  SolverCG<LinearAlgebra::distributed::Vector<double> > solver(solver_control);

  DiagonalMatrix<LinearAlgebra::distributed::Vector<double> > diag_mat;
  matrix_free->initialize_dof_vector(diag_mat.get_vector());
  output = 1.;
  mass_operator.vmult(diag_mat.get_vector(), output);
  for (unsigned int i=0; i<diag_mat.get_vector().local_size(); ++i)
    diag_mat.get_vector().local_element(i) = 1./diag_mat.get_vector().local_element(i);
  if (short_output == false)
    {
      const double diag_norm = diag_mat.get_vector().l2_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Norm of diagonal for preconditioner: " << diag_norm << std::endl;
    }

  double solver_time = 1e10;
  for (unsigned int t=0; t<2; ++t)
    {
      output = 0;
      time.restart();
      try
        {
          solver.solve(mass_operator, output, input, diag_mat);
        }
      catch (SolverControl::NoConvergence &e)
        {
          // prevent the solver to throw an exception in case we should need more
          // than 100 iterations
        }
      data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      solver_time = std::min(data.max, solver_time);
    }

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

  double matvec_time = 1e10;
  for (unsigned int t=0; t<2; ++t)
    {
      time.restart();
      for (unsigned int i=0; i<50; ++i)
        mass_operator.vmult(input, output);
      data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      matvec_time = std::min(data.max/50, matvec_time);
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::setw(2) << fe_degree << " | " << std::setw(2) << n_q_points
              << " | " << std::setw(10) << tria.n_global_active_cells()
              << " | " << std::setw(11) << dof_handler.n_dofs()
              << " | " << std::setw(11) << solver_time/solver_control.last_step()
              << " | " << std::setw(11) << dof_handler.n_dofs()/solver_time*solver_control.last_step()
              << " | " << std::setw(6) << solver_control.last_step()
              << " | " << std::setw(11) << matvec_time
              << std::endl;
}



template <int dim, int fe_degree, int n_q_points>
void do_test(const int s_in,
             const bool compact_output)
{
  if (s_in < 1)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << " p |  q | n_elements |      n_dofs |     time/it |   dofs/s/it | CG_its | time/matvec"
                  << std::endl;
      unsigned int s =
        std::max(3U, static_cast<unsigned int>
                 (std::log2(1024/fe_degree/fe_degree/fe_degree)));
      while ((8+Utilities::fixed_power<dim>(fe_degree+1))*(1UL<<s)
             < 6000000ULL*Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        {
          test<dim,fe_degree,n_q_points>(s, compact_output);
          ++s;
        }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << std::endl << std::endl;
    }
  else
    test<dim,fe_degree,n_q_points>(s_in, compact_output);
}



int main(int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  unsigned int degree = 1;
  unsigned int s = -1;
  bool compact_output = true;
  if (argc > 1)
    degree = std::atoi(argv[1]);
  if (argc > 2)
    s = std::atoi(argv[2]);
  if (argc > 3)
    compact_output = std::atoi(argv[3]);

  if (degree == 1)
    do_test<3,1,3>(s, compact_output);
  else if (degree == 2)
    do_test<3,2,4>(s, compact_output);
  else if (degree == 3)
    do_test<3,3,5>(s, compact_output);
  else if (degree == 4)
    do_test<3,4,6>(s, compact_output);
  else if (degree == 5)
    do_test<3,5,7>(s, compact_output);
  else if (degree == 6)
    do_test<3,6,8>(s, compact_output);
  else if (degree == 7)
    do_test<3,7,9>(s, compact_output);
  else if (degree == 8)
    do_test<3,8,10>(s, compact_output);
  else if (degree == 9)
    do_test<3,9,11>(s, compact_output);
  else if (degree == 10)
    do_test<3,10,12>(s, compact_output);
  else if (degree == 11)
    do_test<3,11,13>(s, compact_output);
  else
    AssertThrow(false, ExcMessage("Only degrees up to 11 implemented"));

  return 0;
}
