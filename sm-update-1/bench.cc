#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi_compute_index_owner_internal.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_sm_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <chrono>
#include <vector>

#include "../sm-util/mpi.h"
#include "../sm-util/timers.h"

using namespace dealii;

using Number = double;

const MPI_Comm comm = MPI_COMM_WORLD;

template <int dim>
void
test(ConvergenceTable & table,
     const MPI_Comm     comm,
     const MPI_Comm     comm_sm,
     const unsigned int degree,
     const unsigned int n_refinements)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

  // 1) create Triangulation
  parallel::distributed::Triangulation<dim> tria(comm);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_refinements);

  // 2) create DoFHandler
  DoFHandler<dim> dof_handler(tria);
  FE_Q<dim>       fe(degree);
  dof_handler.distribute_dofs(fe);

  // 3) create MatrixFree
  AffineConstraints<Number> constraint;
  constraint.close();

  QGauss<1> quad(degree + 1);

  MappingQGeneric<dim> mapping(1);

  typename dealii::MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_gradients | update_JxW_values | update_quadrature_points;

  dealii::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraint, quad, additional_data);

  // 4) create shared-memory Partitioner and Vector
  LinearAlgebra::SharedMPI::Vector<Number> vec_sm;
  matrix_free.initialize_dof_vector(vec_sm, comm_sm);

  // 5) create distributed Vector
  LinearAlgebra::distributed::Vector<Number> vec;
  matrix_free.initialize_dof_vector(vec);

  // 6) run performance tests
  auto run = [&](const auto &label, const auto &runnable) {
    const unsigned int n_repertitions = 1000;

    MPI_Barrier(comm);
    const auto temp = std::chrono::system_clock::now();
    MPI_Barrier(comm);
    {
      hyperdeal::ScopedLikwidTimerWrapper likwid(label);
      for (unsigned int i = 0; i < n_repertitions; i++)
        runnable();
    }
    MPI_Barrier(comm);

    const auto ms =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - temp)
        .count();

    const auto result =
      static_cast<double>(Utilities::MPI::sum(vec.get_partitioner()->n_ghost_indices(), comm)) *
      sizeof(Number) * n_repertitions / ms / 1000;

    table.add_value(label, result);
    table.set_scientific(label, true);
    return result;
  };

  table.add_value("size_local", Utilities::MPI::sum(vec.get_partitioner()->local_size(), comm));
  table.add_value("size_ghost",
                  Utilities::MPI::sum(vec.get_partitioner()->n_ghost_indices(), comm));

  // clang-format off
  
  // ... update ghost values
  table.add_value("ghost::speedup", 
      run("ghost::L::S::V", [&]() { vec_sm.update_ghost_values(); }) /  
      run("ghost::L::D::V", [&]() { vec.update_ghost_values(); }));
  table.set_scientific("ghost::speedup", true);
  
  // ... compress
  table.add_value("compress::speedup", 
      run("compress::L::S::V", [&]() { vec_sm.compress(VectorOperation::values::add); }) /  
      run("compress::L::D::V", [&]() { vec.compress(VectorOperation::values::add); }));
      
  table.set_scientific("compress::speedup", true);
  // clang-format on
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int rank = Utilities::MPI::this_mpi_process(comm);

  ConditionalOStream pcout(std::cout, rank == 0);

  // print help page if requested
  if (std::string(argv[1]) == "--help")
    {
      pcout << "mpirun -np 40 ./bench group_size dim degree n_refinements" << std::endl;
      return 0;
    }

  // read parameters form command line
  const unsigned int group_size    = argc < 2 ? 1 : atoi(argv[1]);
  const unsigned int dim           = argc < 3 ? 3 : atoi(argv[2]);
  const unsigned int degree        = argc < 4 ? 3 : atoi(argv[3]);
  const unsigned int n_refinements = argc < 5 ? 8 : atoi(argv[4]);

  hyperdeal::ScopedLikwidInitFinalize likwid;

  // create convergence table
  ConvergenceTable table;

  // run tests for different group sizes
  for (auto size : (group_size == 0 ? hyperdeal::mpi::create_possible_group_sizes(comm) :
                                      std::vector<unsigned int>{group_size}))
    {
      table.add_value("group_size", size);

      // create sm communicator
      MPI_Comm comm_sm;
      MPI_Comm_split(comm, rank / size, rank, &comm_sm);

      // perform tests
      if (dim == 2)
        test<2>(table, comm, comm_sm, degree, n_refinements);
      else if (dim == 3)
        test<3>(table, comm, comm_sm, degree, n_refinements);

      // free communicator
      MPI_Comm_free(&comm_sm);
    }

  // print convergence table
  if (pcout.is_active())
    table.write_text(pcout.get_stream());
}