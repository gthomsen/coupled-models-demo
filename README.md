# Overview

Demonstration of coupling solver and particle tracker models via MPI.  This
illustrates the benefits of separating each model into separate applications:

 * Development, testing, and validation can be done on each model independently,
   and likely, in greater detail.

 * Different solvers and particle trackers can be coupled together so long as
   they adhere to a basic interface of transmitting flow variables.  This
   can be done to explore different techniques (replace the piece you're
   interested in) or take advantage of computational resources (implement one or
   the other to utilize hardware accelerators like GPUs).

 * Perform basic performance benchmarks before committing to an implementation
   or target system.

 * Gives a roadmap of necessary MPI machinery in Python before translating it
   into a less flexible/higher performance language.

# Installation

Requires a recent version of `mpi4py`.

``` shell
# or use Conda.
$ pip install mpi4py
```

Should run in both Python 2 and Python 3, though only tested with the latter.

# Execution
Pick a number of a number of grid points and particles, as well as the number of
solvers and particle trackers, then run both.

A simple, single-node command line MPMD approach is the following:

``` shell
$ export GRID_POINTS=1000000 NUMBER_PARTICLES=100000
$ export NUMBER_SOLVERS=4 NUMBER_TRACKERS=2

# executes 5 iterations of both applications; 3 seconds per timestep for
# the solver and 1 second per timestep for the particle tracker.
$ mpirun -np ${NUMBER_SOLVERS} python ./solver.py 3 5 ${GRID_POINTS} \
       : -np ${NUMBER_TRACKERS} python ./tracker.py 1 5 ${GRID_POINTS} ${NUMBER_PARTICLES}

```

More sophisticated setup so that solver and particle tracker ranks can be
carefully positioned across nodes should be done for meaningful benchmarks.
This requires configurations that are MPI distribution-specific and knowledge
of the target machine's architecture.

# Architecture

The current implementation is intended to be used as learning tools so as to
guide updates to more complex models.  Rather than carefully implement each
aspect of a solver and a particle tracker, the major components needed are
parameters so the interaction can be demonstrated.

Overview of the execution flow:

 1. Get configuration for each application.
 2. Partition the MPI universe into potentially two applications: 1) solver and
    2) tracker.  The tracker exits if a solver is not detected.
 3. Validate the solver's configuration and synchronize across all ranks, both
    solvers and particle trackers, exiting early if misconfiguration is detected.
 4. If the tracker is executing validate its configuration and synchronize it
    to the solvers, exiting early if a misconfiguration is detected.
 5. Simulate the timesteps requested, transferring velocities from the solver to
    the particle tracker after they have been computed.

The entire grid is transferred to each tracker to minimize the coupling
between the solver and the tracker.  Should this become untenable due to memory
resources on the individual trackers, this can be improved with a little bit
of thought and effort.

## Potential Improvements

This was initially written as a proof-of-concept to show how straight forward
decoupled solvers/particle trackers would be.  While implementing additional
ideas popped up, both for immediate implementation and for higher performance
optimizations.

Things to explore include:

 1. Structuring the grid so that custom MPI datatypes can be used to transfer
    fragments of the grid as a single collective rather than as multiple
    collectives.  Currently, there are three `MPI_Allgather()` calls for each
    time step which do not scale well with large node counts.

 2. Using MPI-3's non-blocking collectives (NBCs) to transfer grid variables to
    the particle trackers while computing the next timestep's values.  This would
    involve `MPI_IAllgather()` and adding logic to ensure that the update of grid
    variables due to time stepping doesn't interfere with an in-progress
    transfer.  Presumably the time stepping scheme ensures memory is available
    for the field variables transferred and reduces the problem to finding
    resources to make progress in the background while computations occur.

 3. Using a more sophisticated distribution scheme of the grid variables.
    Currently the solver's ranks send their portion of the grid variables to each
    particle tracker so that the trackers have the full grid.  While this is
    potentially a naive approach (it's inherently memory limited when time
    stepping schemes are considered) it is a lower bound on performance if
    portions of the grid need to be distributed to individual trackers.  Care
    would need to be taken by the trackers to communicate which ranks needed
    which subset of the grid for dynamic load balancing.

## Todo

The utility of these codes could be improved by the following:

 * Report the memory footprints required, in aggregate and per rank, for the
   parameters requested.
