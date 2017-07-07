
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

using namespace dealii;

// small deformation in mesh to avoid triggering the constant Jacobian case
template <int dim>
class MyManifold : public ChartManifold<dim,dim,dim>
{
public:
  MyManifold() : factor(0.1) {}

  virtual Point<dim> push_forward(const Point<dim> &p) const
  {
    double sinval = factor;
    for (unsigned int d=0; d<dim; ++d)
      sinval *= std::sin(numbers::PI*p[d]);
    Point<dim> out;
    for (unsigned int d=0; d<dim; ++d)
      out[d] = p[d] + sinval;
    return out;
  }

  virtual Point<dim> pull_back(const Point<dim> &p) const
  {
    Point<dim> x = p;
    Point<dim> one;
    for (unsigned int d=0; d<dim; ++d)
      one(d) = 1.;

    // Newton iteration to solve the nonlinear equation given by the point
    Tensor<1,dim> sinvals;
    for (unsigned int d=0; d<dim; ++d)
      sinvals[d] = std::sin(numbers::PI*x(d));

    double sinval = factor;
    for (unsigned int d=0; d<dim; ++d)
      sinval *= sinvals[d];
    Tensor<1,dim> residual = p - x - sinval*one;
    unsigned int its = 0;
    while (residual.norm() > 1e-12 && its < 100)
      {
        Tensor<2,dim> jacobian;
        for (unsigned int d=0; d<dim; ++d)
          jacobian[d][d] = 1.;
        for (unsigned int d=0; d<dim; ++d)
          {
            double sinval_der = factor * numbers::PI * std::cos(numbers::PI*x(d));
            for (unsigned int e=0; e<dim; ++e)
              if (e!=d)
                sinval_der *= sinvals[e];
            for (unsigned int e=0; e<dim; ++e)
              jacobian[e][d] += sinval_der;
          }

        x += invert(jacobian) * residual;

        for (unsigned int d=0; d<dim; ++d)
          sinvals[d] = std::sin(numbers::PI*x(d));

        sinval = factor;
        for (unsigned int d=0; d<dim; ++d)
          sinval *= sinvals[d];
        residual = p - x - sinval*one;
        ++its;
      }
    AssertThrow (residual.norm() < 1e-12,
                 ExcMessage("Newton for point did not converge."));
    return x;
  }

private:
  const double factor;
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

  FE_Q<dim> fe(fe_degree);
  MappingQGeneric<dim> mapping(fe_degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  ConstraintMatrix constraints;
  constraints.close();
  std::shared_ptr<MatrixFree<dim,double> > matrix_free(new MatrixFree<dim,double>());
  matrix_free->reinit(mapping, dof_handler, constraints, QGauss<1>(n_q_points),
                      typename MatrixFree<dim,double>::AdditionalData());

  MatrixFreeOperators::MassOperator<dim,fe_degree,n_q_points> mass_operator;
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

  ReductionControl solver_control(1000, 1e-15, 1e-6);
  SolverCG<LinearAlgebra::distributed::Vector<double> > solver(solver_control);

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
              << " | " << std::setw(11) << dof_handler.n_dofs()
              << " | " << std::setw(11) << solver_time/solver_control.last_step()
              << " | " << std::setw(11) << dof_handler.n_dofs()/solver_time*solver_control.last_step()
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
  while (Utilities::fixed_power<dim>(fe_degree+1)*(1UL<<s)
         < 3000000ULL*Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
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
