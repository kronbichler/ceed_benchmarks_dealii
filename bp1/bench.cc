
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>
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
#include "../common_code/solver_cg_optimized.h"
#include "../common_code/renumber_dofs_for_mf.h"


#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif

using namespace dealii;

// VERSION:
//   0: p:d:t + vectorized over elements; 
//   1: p:f:t + vectorized within element (not optimized)
#define VERSION 1 

#if VERSION == 0
  typedef dealii::VectorizedArray<double> VectorizedArrayType;
#elif VERSION == 1
  typedef dealii::VectorizedArray<double, 1> VectorizedArrayType;
#endif


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
  MappingQGeneric<dim> mapping(std::min(fe_degree, 6));
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  constraints.close();
  typename MatrixFree<dim,double, VectorizedArrayType>::AdditionalData mf_data;

  // renumber Dofs to minimize the number of partitions in import indices of
  // partitioner
  renumber_dofs_mf<dim,double, VectorizedArrayType>(dof_handler, constraints, mf_data);

  std::shared_ptr<MatrixFree<dim,double, VectorizedArrayType> > matrix_free(new MatrixFree<dim,double, VectorizedArrayType>());
  matrix_free->reinit(mapping, dof_handler, constraints, QGauss<1>(n_q_points),
                      mf_data);

  if (false)
    for (unsigned int p=0; p<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++p)
      {
        if (p == Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
          {
            std::cout << "zero list: ";
            for (unsigned int i=0; i<matrix_free->get_dof_info().vector_zero_range_list_index.size()-1; ++i)
              {
                for (unsigned int j=matrix_free->get_dof_info().vector_zero_range_list_index[i];
                     j<matrix_free->get_dof_info().vector_zero_range_list_index[i+1]; ++j)
                  {
                    std::cout << matrix_free->get_dof_info().vector_zero_range_list[j].first << " "
                              << matrix_free->get_dof_info().vector_zero_range_list[j].second << "   ";
                  }
                std::cout << std::endl;
              }

            std::cout << "pre list: ";
            for (unsigned int i=0; i<matrix_free->get_dof_info().cell_loop_pre_list_index.size()-1; ++i)
              {
                for (unsigned int j=matrix_free->get_dof_info().cell_loop_pre_list_index[i];
                     j<matrix_free->get_dof_info().cell_loop_pre_list_index[i+1]; ++j)
                  {
                    std::cout << matrix_free->get_dof_info().cell_loop_pre_list[j].first << " "
                              << matrix_free->get_dof_info().cell_loop_pre_list[j].second << "   ";
                  }
                std::cout << std::endl;
              }

            std::cout << "post list: ";
            for (unsigned int i=0; i<matrix_free->get_dof_info().cell_loop_post_list_index.size()-1; ++i)
              {
                for (unsigned int j=matrix_free->get_dof_info().cell_loop_post_list_index[i];
                     j<matrix_free->get_dof_info().cell_loop_post_list_index[i+1]; ++j)
                  {
                    std::cout << matrix_free->get_dof_info().cell_loop_post_list[j].first << " "
                              << matrix_free->get_dof_info().cell_loop_post_list[j].second << "   ";
                  }
                std::cout << std::endl;
              }
          }
        usleep(10000);
        MPI_Barrier(MPI_COMM_WORLD);
      }

  Mass::MassOperator<dim,fe_degree,n_q_points, 1, double, VectorizedArrayType> mass_operator;
  mass_operator.initialize(matrix_free, constraints);

  LinearAlgebra::distributed::Vector<double> input, output, tmp;
  matrix_free->initialize_dof_vector(input);
  matrix_free->initialize_dof_vector(output);
  matrix_free->initialize_dof_vector(tmp);
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

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("cg_solver");
#endif
  double solver_time = 1e10;
  for (unsigned int t=0; t<4; ++t)
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
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("cg_solver");
#endif
  if (short_output==false)
    {
      tmp.equ(-1.0, input);
      mass_operator.vmult_add(tmp, output);
      deallog << "True residual norm: " << tmp.l2_norm() << std::endl;
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

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("cg_solver_opt");
#endif
  SolverCGOptimized<LinearAlgebra::distributed::Vector<double> > solver2(solver_control);
  double solver_time2 = 1e10;
  for (unsigned int t=0; t<4; ++t)
    {
      output = 0;
      time.restart();
      try
        {
          solver2.solve(mass_operator, output, input, diag_mat);
        }
      catch (SolverControl::NoConvergence &e)
        {
          // prevent the solver to throw an exception in case we should need more
          // than 100 iterations
        }
      data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      solver_time2 = std::min(data.max, solver_time2);
    }
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("cg_solver_opt");
#endif
  if (short_output==false)
    {
      tmp.equ(-1.0, input);
      mass_operator.vmult_add(tmp, output);
      deallog << "True residual norm: " << tmp.l2_norm() << std::endl;
    }
  const unsigned int solver_its2 = solver_control.last_step();

  SolverCGOptimizedAllreduce<LinearAlgebra::distributed::Vector<double> > solver3(solver_control);
  double solver_time3 = 1e10;
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("cg_solver_optv");
#endif
  for (unsigned int t=0; t<4; ++t)
    {
      output = 0;
      time.restart();
      try
        {
          solver3.solve(mass_operator, output, input, diag_mat);
        }
      catch (SolverControl::NoConvergence &e)
        {
          // prevent the solver to throw an exception in case we should need more
          // than 100 iterations
        }
      data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      solver_time3 = std::min(data.max, solver_time3);
    }
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("cg_solver_optv");
#endif

  if (short_output==false)
    {
      tmp.equ(-1.0, input);
      mass_operator.vmult_add(tmp, output);
      deallog << "True residual norm: " << tmp.l2_norm() << std::endl;
    }

  SolverCGFullMerge<LinearAlgebra::distributed::Vector<double> > solver4(solver_control);
  double solver_time4 = 1e10;
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("cg_solver_optm");
#endif
  for (unsigned int t=0; t<4; ++t)
    {
      output = 0;
      time.restart();
      try
        {
          solver4.solve(mass_operator, output, input, diag_mat);
        }
      catch (SolverControl::NoConvergence &e)
        {
          // prevent the solver to throw an exception in case we should need more
          // than 100 iterations
        }
      data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      solver_time4 = std::min(data.max, solver_time4);
    }
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("cg_solver_optm");
#endif

  if (short_output==false)
    {
      tmp.equ(-1.0, input);
      mass_operator.vmult_add(tmp, output);
      deallog << "True residual norm: " << tmp.l2_norm() << std::endl;
    }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("matvec");
#endif
  double matvec_time = 1e10;
  for (unsigned int t=0; t<4; ++t)
    {
      time.restart();
      for (unsigned int i=0; i<50; ++i)
        mass_operator.vmult(tmp, output);
      data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      matvec_time = std::min(data.max/50, matvec_time);
    }
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("matvec");
#endif

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::setw(2) << fe_degree << " | " << std::setw(2) << n_q_points
              << " | " << std::setw(10) << tria.n_global_active_cells()
              << " | " << std::setw(11) << dof_handler.n_dofs()
              << " | " << std::setw(11) << solver_time/solver_control.last_step()
              << " | " << std::setw(11) << dof_handler.n_dofs()/solver_time2*solver_control.last_step()
              << " | " << std::setw(11) << solver_time2/solver_control.last_step()
              << " | " << std::setw(11) << solver_time3/solver_control.last_step()
              << " | " << std::setw(11) << solver_time4/solver_control.last_step()
              << " |"  << std::setw(3) << solver_its2
              << " "   << std::setw(3) << solver_control.last_step()
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
        std::cout << " p |  q | n_elements |      n_dofs |     time/it |op_dofs/s/it | opt_time/it | op3_time/it | opm_time/it | CG_its | time/matvec"
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
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#endif

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
  else if (degree == 12)
    do_test<3,12,14>(s, compact_output);
  else if (degree == 13)
    do_test<3,13,15>(s, compact_output);
  else if (degree == 14)
    do_test<3,14,16>(s, compact_output);
  else if (degree == 15)
    do_test<3,15,17>(s, compact_output);
  else if (degree == 16)
    do_test<3,16,18>(s, compact_output);
  else
    AssertThrow(false, ExcMessage("Only degrees up to 16 implemented"));

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
