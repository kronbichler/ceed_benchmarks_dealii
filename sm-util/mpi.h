// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the hyper.deal authors
//
// This file is part of the hyper.deal library.
//
// The hyper.deal library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hyper.deal.
//
// ---------------------------------------------------------------------

#ifndef HYPERDEAL_MPI
#define HYPERDEAL_MPI

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

#endif