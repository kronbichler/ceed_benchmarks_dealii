#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi_compute_index_owner_internal.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_sm_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <chrono>
#include <vector>

using namespace dealii;

using Number              = double;
using VectorizedArrayType = VectorizedArray<Number>;

const MPI_Comm comm = MPI_COMM_WORLD;

template <int dim, int degree, int n_points_1d = degree + 1>
void
test(ConvergenceTable & table,
     const MPI_Comm     comm,
     const MPI_Comm     comm_sm,
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

  QGauss<1> quad(n_points_1d);

  MappingQGeneric<dim> mapping(1);

  typename dealii::MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_gradients | update_JxW_values | update_quadrature_points;
  additional_data.overlap_communication_computation = true;

  dealii::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraint, quad, additional_data);

  // 4) run performance tests
  auto run = [&](const auto &label, const auto &runnable) {
    const unsigned int n_repertitions = 1000;

    MPI_Barrier(comm);
    const auto temp = std::chrono::system_clock::now();
    MPI_Barrier(comm);
    for (unsigned int i = 0; i < n_repertitions; i++)
      runnable();
    MPI_Barrier(comm);

    const auto ms =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - temp)
        .count();

    table.add_value(label,
                    static_cast<double>(dof_handler.n_dofs()) * sizeof(Number) * n_repertitions /
                      ms / 1000);
    table.set_scientific(label, true);
  };

  // .. for LinearAlgebra::distributed::Vector
  {
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    // create vectors
    VectorType vec1, vec2;
    matrix_free.initialize_dof_vector(vec1);
    matrix_free.initialize_dof_vector(vec2);

    table.add_value("size_local", Utilities::MPI::sum(vec1.get_partitioner()->local_size(), comm));
    table.add_value("size_ghost",
                    Utilities::MPI::sum(vec1.get_partitioner()->n_ghost_indices(), comm));

    // apply mass matrix
    run("L::D::V", [&]() {
      FEEvaluation<dim, degree, n_points_1d, 1, Number, VectorizedArrayType> phi(matrix_free);
      matrix_free.template cell_loop<VectorType, VectorType>(
        [&](const auto &, auto &dst, const auto &src, const auto cells) {
          for (unsigned int cell = cells.first; cell < cells.second; cell++)
            {
              phi.reinit(cell);
              phi.gather_evaluate(src, true, false);

              for (unsigned int q = 0; q < phi.n_q_points; q++)
                phi.submit_value(phi.get_value(q), q);

              phi.integrate_scatter(true, false, dst);
            }
        },
        vec1,
        vec2);
    });
  }

  // .. for LinearAlgebra::SharedMPI::Vector
  {
    using VectorType = LinearAlgebra::SharedMPI::Vector<Number>;

    // create vectors
    VectorType vec1, vec2;
    matrix_free.initialize_dof_vector(vec1, comm_sm);
    matrix_free.initialize_dof_vector(vec2, comm_sm);

    // apply mass matrix
    run("L::S::V", [&]() {
      FEEvaluation<dim, degree, n_points_1d, 1, Number, VectorizedArrayType> phi(matrix_free);
      matrix_free.template cell_loop<VectorType, VectorType>(
        [&](const auto &, auto &dst, const auto &src, const auto cells) {
          for (unsigned int cell = cells.first; cell < cells.second; cell++)
            {
              phi.reinit(cell);
              phi.gather_evaluate(src, true, false);

              for (unsigned int q = 0; q < phi.n_q_points; q++)
                phi.submit_value(phi.get_value(q), q);

              phi.integrate_scatter(true, false, dst);
            }
        },
        vec1,
        vec2);
    });
  }
}

template <int dim>
void
test_dim(ConvergenceTable & table,
         const MPI_Comm     comm,
         const MPI_Comm     comm_sm,
         const unsigned int degree,
         const unsigned int n_refinements)
{
  switch (degree)
    {
      case 3:
        test<dim, 3>(table, comm, comm_sm, n_refinements);
        break;
      default:
        Assert(false, ExcNotImplemented());
    }
}

namespace hyperdeal
{
  namespace mpi
  {
    MPI_Comm
    create_sm(const MPI_Comm &comm)
    {
      int rank;
      MPI_Comm_rank(comm, &rank);

      MPI_Comm comm_shared;
      MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &comm_shared);

      return comm_shared;
    }



    unsigned int
    n_procs_of_sm(const MPI_Comm &comm, const MPI_Comm &comm_sm)
    {
      // determine size of current shared memory communicator
      int size_shared;
      MPI_Comm_size(comm_sm, &size_shared);

      // determine maximum, since some shared memory communicators might not be
      // filed completely
      int size_shared_max;
      MPI_Allreduce(&size_shared, &size_shared_max, 1, MPI_INT, MPI_MAX, comm);

      return size_shared_max;
    }



    std::vector<unsigned int>
    create_possible_group_sizes(const MPI_Comm &comm)
    {
      MPI_Comm           comm_sm       = hyperdeal::mpi::create_sm(comm);
      const unsigned int n_procs_of_sm = hyperdeal::mpi::n_procs_of_sm(comm, comm_sm);
      MPI_Comm_free(&comm_sm);

      std::vector<unsigned int> result;

      for (unsigned int i = 1; i <= n_procs_of_sm; i++)
        if (n_procs_of_sm % i == 0)
          result.push_back(i);

      return result;
    }
  } // namespace mpi
} // namespace hyperdeal

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
        test_dim<2>(table, comm, comm_sm, degree, n_refinements);
      else if (dim == 3)
        test_dim<3>(table, comm, comm_sm, degree, n_refinements);

      // free communicator
      MPI_Comm_free(&comm_sm);
    }

  // print convergence table
  if (pcout.is_active())
    table.write_text(pcout.get_stream());
}