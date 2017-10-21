#/usr/bin/env python

from __future__ import print_function

from mpi4py import MPI
import MPIUtility
import numpy as np
import sys
import time

# toy model of a particle tracker that can be run coupled to a solver.  detects
# how it is invoked and sets up the appropriate communicators needed.  if it is
# run on it's own it shuts down as this toy does not support a standalone
# configuration.  the number of grid points and particles can be specified to
# do basic benchmarking of transfer times and memory requirements.
#
# see the solver for a list of potential improvements.

def configure_tracker( application_rank, application_size, number_particles ):
    """
    Configures the tracker and returns the number of errors encountered while
    doing so.

    Takes 3 arguments:

      application_rank - Rank of this tracker.
      application_size - Number of tracker ranks executing.
      number_particles - Number of particles requested for tracking.

    Returns 1 value:

      error_count - Number of errors encountered during configuration.  0
                    indicates successful configuration.

    """

    error_count = 0

    if number_particles <= 0:
        if application_rank == 0:
            print( "No particles specified!" )
    # NOTE: we include this configuration check only so we have a knob to turn
    #       that can induce failure at startup.
    elif (number_particles % application_size) != 0:
        if application_rank == 0:
            print( "Number of particles ({:d}) must be divisible by number of ranks ({:d})!".format( number_particles,
                                                                                                     application_size ) )
        error_count += 1

    return error_count

if len( sys.argv ) != 5:
    print( "Usage: {:s} <delay> <iterations> <grid points> <particles>".format( sys.argv[0] ) )
    sys.exit( 1 )

delay_length       = float( sys.argv[1] )
number_iterations  = int( sys.argv[2] )
number_grid_points = int( sys.argv[3] )
number_particles   = int( sys.argv[4] )

world_comm = MPI.COMM_WORLD
world_rank = world_comm.Get_rank()
world_size = world_comm.Get_size()

startup_error_count = 0

# get the characteristics of the trackers.
application_comm = MPIUtility.partition_by_string( world_comm, sys.argv[0] )
application_rank = application_comm.Get_rank()
application_size = application_comm.Get_size()

# determine if there are solvers present in the universe by comparing our newly
# split communicator against the world.
#
# NOTE: we check against unequality to simplify the logic.  otherwise we have to
#       check against similar/congruent/identical communicators.
coupled_flag = (MPI.Comm.Compare( application_comm, world_comm ) == MPI.UNEQUAL)

if not coupled_flag:
    # we're being run standalone.  if coupling was optional this is where
    # we'd setup our configuration, though we simply exit since we're
    # only demonstrating the coupled model approach.
    if world_rank == 0:
        print( "Looks like we're all alone in the universe.  Exiting." )
    sys.exit( 1 )

# check to see if the solver started up and exit if not.
startup_error_count = world_comm.bcast( 0, 0 )

if startup_error_count > 0:
    sys.exit( 1 )

# configure the tracker and confirm our parameters make sense.  we broadcast
# the failure count so the solver can shutdown if we're not able to run.
startup_error_count = configure_tracker( application_rank,
                                         application_size,
                                         number_particles )

# create an intercommunicator to interact with the tracker via the world
# intracommunicator (which contains both solver and tracker).
#
# NOTE: the leader in the remote group is the first rank in the universe.
#
intercomm = application_comm.Create_intercomm( 0, world_comm, 0 )

# send our startup status to the solver.
MPIUtility.intercomm_broadcast( intercomm,
                                startup_error_count,
                                application_rank,
                                0,
                                True )

if startup_error_count > 0:
    sys.exit( 1 )

# each tracker has space for the entire grid.  velocities for each grid point
# are received by each process so the particles can be efficiently load
# balanced.  this may be distributing particles amongst the trackers or
# distributing a subset of the grid to individual trackers for truly
# large grids.
velocity_x = np.empty( number_grid_points, dtype="d" )
velocity_y = np.empty( number_grid_points, dtype="d" )
velocity_z = np.empty( number_grid_points, dtype="d" )
empty_data = np.zeros( 0, dtype="d" )

for iteration_count in range( number_iterations ):

    # send nothing, receive things.
    #
    # XXX: a real implementation should have a custom MPI data type that
    #      ensures each solver's grid fragment is sent as a single block of
    #      data rather than as three.
    #
    # XXX: transfer of previous velocities could be done while the current
    #      are being computed via IAllgather().
    intercomm.Allgather( [empty_data, MPI.DOUBLE],
                         [velocity_x, MPI.DOUBLE] )

    intercomm.Allgather( [empty_data, MPI.DOUBLE],
                         [velocity_y, MPI.DOUBLE] )

    intercomm.Allgather( [empty_data, MPI.DOUBLE],
                         [velocity_z, MPI.DOUBLE] )

    print( "Tracking particles [tracker {:d}].".format( world_rank ) )
    time.sleep( delay_length )

MPI.Finalize()
