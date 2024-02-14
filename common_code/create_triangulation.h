#ifndef create_triangulation_h_
#define create_triangulation_h_

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/grid/tria_description.h>

template <int dim, int spacedim = dim>
std::shared_ptr<parallel::TriangulationBase<dim, spacedim>>
create_triangulation(const unsigned int s, MyManifold<dim> &manifold, const bool use_pft = false)
{
  std::shared_ptr<parallel::TriangulationBase<dim, spacedim>> tria_shared;

  const unsigned int        n_refine  = s / 12;
  const unsigned int        remainder = s % 12;
  std::vector<unsigned int> subdivisions(dim, 1);
  subdivisions[dim - 1] = 4 + remainder % 4;
  if (remainder > 3)
    subdivisions[1] = 2;
  if (remainder > 7)
    subdivisions[0] = 2;

  Point<dim> p2;
  for (unsigned int d = 0; d < dim; ++d)
    p2[d] = subdivisions[d];

  if (use_pft == false)
    {
      auto tria = new parallel::distributed::Triangulation<dim>(MPI_COMM_WORLD);
      GridGenerator::subdivided_hyper_rectangle(*tria, subdivisions, Point<dim>(), p2);
      GridTools::transform(
        std::bind(&MyManifold<dim>::push_forward, manifold, std::placeholders::_1), *tria);
      tria->set_all_manifold_ids(1);
      tria->set_manifold(1, manifold);
      tria->refine_global(n_refine);

      tria_shared.reset(tria);
    }
  else
    {
      dealii::Triangulation<dim> tria_serial;
      GridGenerator::subdivided_hyper_rectangle(tria_serial, subdivisions, Point<dim>(), p2);
      GridTools::transform(
        std::bind(&MyManifold<dim>::push_forward, manifold, std::placeholders::_1), tria_serial);
      tria_serial.set_all_manifold_ids(1);
      tria_serial.set_manifold(1, manifold);
      tria_serial.refine_global(n_refine);

      {
        const unsigned int n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
        const unsigned int n_cells_per_process =
          (tria_serial.n_active_cells() + n_procs - 1) / n_procs;

        // partition equally the active cells
        for (auto cell : tria_serial.active_cell_iterators())
          cell->set_subdomain_id(cell->active_cell_index() / n_cells_per_process);
      }

      const auto cd =
        TriangulationDescription::Utilities::create_description_from_triangulation(tria_serial,
                                                                                   MPI_COMM_WORLD);

      auto tria = new parallel::fullydistributed::Triangulation<dim>(MPI_COMM_WORLD);
      tria->create_triangulation(cd);

      tria_shared.reset(tria);
    }

  return tria_shared;
}

#endif
