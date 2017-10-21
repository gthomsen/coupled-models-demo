#/usr/bin/env python

from __future__ import print_function

from mpi4py import MPI
import MPIUtility
import numpy as np
import sys
import time

# toy model of a solver that can be coupled to a particle tracker.  detects how
# it is invoked and sets up the appropriate communicators needed to transfer
# grid velocities at each time step if particle trackers are available.  the
# number of grid points can be specified to do basic benchmarking of transfer
# times.
#
# possible improvements:
#
#   1. allocate the grid on each solver rank and use custom MPI data types
#      to coalesce multiple collective transfers into a single per timestep.
#
#   2. use non-blocking collectives to transfer the most recent field velocities
#      while computing the next.  requires MPI-3.

def configure_solver( application_rank, application_size, number_grid_points ):
    """
    Configures the solver and returns the number of errors encountered while
    doing so.

    Takes 3 arguments:

      application_rank   - Rank of this solver.
      application_size   - Number of solver ranks executing.
      number_grid_points - Number of grid points to simulate.

    Returns 1 value:

      error_count - Number of errors encountered during configuration.  0
                    indicates successful configuration.

    """

    error_count = 0

    if number_grid_points <= 0:
        if application_rank == 0:
            print( "No grid points specified!" )
        error_count += 1
    elif (number_grid_points % application_size) != 0:
        if application_rank == 0:
            print( "Number of grid points ({:d}) must be divisible by number of ranks ({:d})!".format( number_grid_points,
                                                                                                       application_size ) )
        error_count += 1

    return error_count

if len( sys.argv ) != 4:
    print( "Usage: {:s} <delay> <iterations> <grid points>".format( sys.argv[0] ) )
    sys.exit( 1 )

delay_length       = float( sys.argv[1] )
number_iterations  = int( sys.argv[2] )
number_grid_points = int( sys.argv[3] )

world_comm = MPI.COMM_WORLD
world_rank = world_comm.Get_rank()
world_size = world_comm.Get_size()

# get the characteristics of the solvers.
application_comm = MPIUtility.partition_by_string( world_comm, sys.argv[0] )
application_rank = application_comm.Get_rank()
application_size = application_comm.Get_size()

# configure the solver and confirm our parameters make sense.  we broadcast the
# failure count so we can gracefully shutdown rather than randomly fail later.
# note that we broadcast to the world rather than our application group so that
# the tracker can shutdown as well instead of hanging.
startup_error_count = configure_solver( application_rank,
                                        application_size,
                                        number_grid_points )
world_comm.bcast( startup_error_count, 0 )

if startup_error_count > 0:
    if application_rank == 0:
        print( "Solver had {:d} error{:s} during startup!  Exiting.".format( startup_error_count,
                                                                             "" if startup_error_count else "s" ) )
    sys.exit( 1 )

# determine if there are trackers present in the universe by comparing our newly
# split communicator against the world.
#
# NOTE: we check against unequality to simplify the logic.  otherwise we have to
#       check against similar/congruent/identical communicators.
coupled_flag = (MPI.Comm.Compare( application_comm, world_comm ) == MPI.UNEQUAL)

# setup our communicator with the tracker if it has been started along side us.
# confirm that it started up without issue and terminate ourselves if it didn't.
if coupled_flag:
    # create an intercommunicator to interact with the tracker via the world
    # intracommunicator (which contains both solver and tracker).
    #
    # NOTE: the leader in the remote group is one past this application's last
    #       rank.
    #
    intercomm = application_comm.Create_intercomm( 0, world_comm, application_size )

    # confirm that the tracker's startup was successful.
    startup_error_count = MPIUtility.intercomm_broadcast( intercomm,
                                                          # NOTE: this isn't used since the solver isn't sending anything.
                                                          0,
                                                            application_rank,
                                                          # remote root.
                                                          0,
                                                          False )

    if startup_error_count > 0:
        if application_rank == 0:
            print( "Tracker had {:d} error{:s} during startup.  Exiting.!".format( startup_error_count,
                                                                                   "" if startup_error_count == 1 else "s" ) )
        sys.exit( 1 )

# let the caller know how we were started.
if application_rank == 0:
    print( "Starting the solver with {:d} rank{:s}".format( application_size,
                                                            "" if application_size == 1 else "s" ),
           end="" )
    if coupled_flag:
        print( " and tracking particles with {:d} rank{:s}.".format( world_size - application_size,
                                                                     "" if (world_size - application_size) == 1 else "s" ) )
    else:
        print( "" )

# allocate space for our field's velocities.  we create three variables to
# better reflect the storage on individual solver ranks.
grid_per_rank = number_grid_points // application_size

velocity_x = np.ones( grid_per_rank, dtype="d" ) * application_rank
velocity_y = np.ones( grid_per_rank, dtype="d" ) * application_rank
velocity_z = np.ones( grid_per_rank, dtype="d" ) * application_rank
empty_data = np.zeros( 0, dtype="d" )

for iteration_count in range( number_iterations ):
    print( "Solving for velocities [solver {:d}].".format( application_rank ) )

    time.sleep( delay_length )

    if coupled_flag:
        # send the field's solution for this timestep to the tracker.  this
        # "gather" sends each solver's slice of the grid to each tracker so
        # they all have a full solution.  this allows dynamic partitioning
        # of the grid to efficiently process the particles.
        #
        # NOTE: for very large grids a smarter distribution scheme could be
        #       performed with Allgatherv() though would require back
        #       communication from the trackers to identify portions of the
        #       grid to pull.

        # XXX: a real implementation should have a custom MPI data type that
        #      ensures each solver's grid fragment is sent as a single block of
        #      data rather than as three.
        #
        # XXX: transfer of previous velocities could be done while the current
        #      are being computed via IAllgather().
        timer_start = time.clock()
        intercomm.Allgather( [velocity_x, MPI.DOUBLE],
                             [empty_data, MPI.DOUBLE] )

        intercomm.Allgather( [velocity_y, MPI.DOUBLE],
                             [empty_data, MPI.DOUBLE] )

        intercomm.Allgather( [velocity_z, MPI.DOUBLE],
                             [empty_data, MPI.DOUBLE] )
        transfer_time = time.clock() - timer_start

        if application_rank == 0:
            print( "   {:.2f}s to transfer solution.".format( transfer_time ) )

MPI.Finalize()
