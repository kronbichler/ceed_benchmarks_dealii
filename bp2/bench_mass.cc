
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

#include "../bp1/curved_manifold.h"

using namespace dealii;

// very similar to bp1/bench_mass.cc but using a system of dim components
// rather than a scalar equation.

// In order to use block vectors that is not contained in
// MatrixFreeOperators::MassOperator of deal.II version 8.5, we manually
// implement the respective class here. It is easy enough to do.

template <int dim, int fe_degree, int n_q_points_1d = fe_degree+1, int n_components = 1, typename Number = double>
  class MassOperator
  {
  public:
    /**
     * Number typedef.
     */
    typedef Number value_type;

    /**
     * size_type needed for preconditioner classes.
     */
    typedef types::global_dof_index size_type;

    /**
     * Constructor.
     */
    MassOperator () {}

    /**
     * Initialize function.
     */
    void initialize(std::shared_ptr<const MatrixFree<dim,Number> > data_)
    {
      this->data = data_;
    }

    /**
     * Initialize function.
     */
    void initialize_dof_vector(LinearAlgebra::distributed::BlockVector<Number> &vec) const
    {
      vec.reinit(dim);
      for (unsigned int d=0; d<dim; ++d)
        data->initialize_dof_vector(vec.block(d));
      vec.collect_sizes();
    }

    /**
     * Matrix-vector multiplication.
     */
    void vmult(LinearAlgebra::distributed::BlockVector<Number> &dst,
               const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
      dst = 0;
      this->data->cell_loop (&MassOperator::local_apply_cell,
                             this, dst, src);
    }

    /**
     * Transpose matrix-vector multiplication. Since the mass matrix is
     * symmetric, it does exactly the same as vmult().
     */
    void Tvmult(LinearAlgebra::distributed::BlockVector<Number> &dst,
                const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
      vmult(dst, src);
    }

  private:
    /**
     * For this operator, there is just a cell contribution.
     */
    void local_apply_cell (const MatrixFree<dim,value_type>            &data,
                           LinearAlgebra::distributed::BlockVector<Number> &dst,
                           const LinearAlgebra::distributed::BlockVector<Number> &src,
                           const std::pair<unsigned int,unsigned int>  &cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> phi(data);
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          phi.reinit (cell);
          phi.read_dof_values(src);
          phi.evaluate (true,false);
          for (unsigned int q=0; q<phi.n_q_points; ++q)
            phi.submit_value (phi.get_value(q), q);
          phi.integrate (true,false);
          phi.distribute_local_to_global (dst);
        }
    }

    std::shared_ptr<const MatrixFree<dim,Number> > data;
  };


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
  GridTools::transform(std::bind(&MyManifold<dim>::push_forward, manifold,
                                 std::placeholders::_1),
                       tria);
  tria.set_all_manifold_ids(1);
  tria.set_manifold(1, manifold);
  tria.refine_global(n_refine);

  FE_Q<dim> fe_q(fe_degree);
  //FESystem<dim> fe(fe_q, dim);
  MappingQGeneric<dim> mapping(fe_degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_q);

  ConstraintMatrix constraints;
  constraints.close();
  std::shared_ptr<MatrixFree<dim,double> > matrix_free(new MatrixFree<dim,double>());
  matrix_free->reinit(mapping, dof_handler, constraints, QGauss<1>(n_q_points),
                      typename MatrixFree<dim,double>::AdditionalData());

  MassOperator<dim,fe_degree,n_q_points,dim> mass_operator;
  mass_operator.initialize(matrix_free);

  LinearAlgebra::distributed::BlockVector<double> input, output;
  mass_operator.initialize_dof_vector(input);
  mass_operator.initialize_dof_vector(output);
  for (unsigned int d=0; d<dim; ++d)
    for (unsigned int i=0; i<input.block(d).local_size(); ++i)
      input.block(d).local_element(i) = (i+d)%8;

  Utilities::MPI::MinMaxAvg data =
    Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output == false)
    std::cout << "Setup time:         "
              << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")" << "s"
              << std::endl;

  ReductionControl solver_control(1000, 1e-15, 1e-6);
  SolverCG<LinearAlgebra::distributed::BlockVector<double> > solver(solver_control);

  time.restart();
  solver.solve(mass_operator, output, input, PreconditionIdentity());
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
    mass_operator.vmult(input, output);
  data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      short_output == true)
    std::cout << std::setw(2) << fe_degree << " | " << std::setw(2) << n_q_points
              << " | " << std::setw(10) << tria.n_global_active_cells()
              << " | " << std::setw(11) << dim*dof_handler.n_dofs()
              << " | " << std::setw(11) << solver_time/solver_control.last_step()
              << " | " << std::setw(11) << dim*dof_handler.n_dofs()/solver_time*solver_control.last_step()
              << " | " << std::setw(6) << solver_control.last_step()
              << " | " << std::setw(11) << data.max/100
              << std::endl;
}


template <int dim, int fe_degree, int n_q_points>
void do_test()
{
  unsigned int s =
    std::max(3U, static_cast<unsigned int>
             (std::log2(1024/fe_degree/fe_degree/fe_degree)));
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << " p |  q | n_elements |      n_dofs |     time/it |   dofs/s/it | CG_its | time/matvec"
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

  do_test<3,1,2>();
  do_test<3,1,3>();
  do_test<3,2,3>();
  do_test<3,2,4>();
  do_test<3,3,5>();
  do_test<3,4,6>();
  do_test<3,5,7>();
  do_test<3,6,8>();
  do_test<3,7,9>();
  do_test<3,8,10>();

  return 0;
}
