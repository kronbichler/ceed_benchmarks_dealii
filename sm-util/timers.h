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

#ifndef HYPERDEAL_TIMERS
#define HYPERDEAL_TIMERS

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

namespace hyperdeal
{
  class ScopedLikwidTimerWrapper
  {
  public:
    ScopedLikwidTimerWrapper(const std::string label)
      : label(label)
    {
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(label.c_str());
#endif
    }

    ~ScopedLikwidTimerWrapper()
    {
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(label.c_str());
#endif
    }

  private:
    const std::string label;
  };



  class ScopedLikwidInitFinalize
  {
  public:
    ScopedLikwidInitFinalize()
    {
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_INIT;
      LIKWID_MARKER_THREADINIT;
#endif
    }

    ~ScopedLikwidInitFinalize()
    {
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_CLOSE;
#endif
    }
  };
} // namespace hyperdeal

#endif