#!/usr/bin/env python

from __future__ import print_function

from mpi4py import MPI
import MPIUtility

def demonstrate_broadcast( intercomm, local_rank, world_rank, group_index ):
    """
    Demonstrates broadcast over an intercommunicator.  Conceptually this is a
    single rank in one group broadcasting a value to all of the ranks in the
    other group, while all other non-broadcasting ranks in the first group
    send/receive nothing.  Broadcasts can go from either group to the other
    within an intercommunicator.

    This function demonstrates 3 things:

      1. Sending from the first group to the second, with the first rank
         in the sender being the root.

      2. Sending from the second group to the second, with the first rank
         in the sender being the root.

      3. Sending from the first group to the second, with the second rank
         in the sender being the root.

    Below are values, for each rank, for: 1) the broadcast root, 2) the sent
    value, and 3) the received value:

                   Global   Local     Root         Sent     Received
                    Rank     Rank     Rank        Value      Value

                     0        0      MPI_ROOT      12345      None
       Group #1      1        1    MPI_PROC_NULL    N/A       None
                     2        2    MPI_PROC_NULL    N/A       None
                                       ...
                     n        n    MPI_PROC_NULL    N/A       None

       Group #2     n+1       0         0           N/A       12345
                    n+2       1         0           N/A       12345
                                       ...
                    n+m-1     m         0           N/A       12345

    NOTE: The root rank does not need to be 0 so long it is specified
          correctly.

    Takes 4 arguments:

      intercomm   - Intercommunicator shared across all of the ranks
                    participating in the collective.
      local_rank  - Rank in the local group attached to the intercommunicator.
      world_rank  - Rank in the MPI universe.  Note that this is only used
                    for printing diagnostics rather than participating in
                    the collective operation.
      group_index - Index specifying which group the local rank is in.  This
                    governs which ranks specify MPI.ROOT/MPI.PROC_NULL and
                    which specify an actual rank for the root rank.

    Returns nothing.

    """

    # print a banner announcing what we're doing.
    if world_rank == 0:
        print( (" ================= Broadcast Across an Intercommunicator =================" +
                "\n") )

    # 1. first group broadcasts to second group.

    local_root    = 0
    sending_group = 0

    if group_index == sending_group and local_rank == local_root:
        print( ("   1. First Group to Second Group\n" +
                "\n" +
                "      All ranks in first group participate in the collective, though do not receive\n" +
                "      the data broadcast by the rank amongst them.  Only the ranks in the second\n" +
                "      group receive the data.\n" +
                "\n" ) )

    # synchronize all of the ranks so our banner is more likely to occur before
    # any output below.
    intercomm.Barrier()

    value = MPIUtility.intercomm_broadcast( intercomm,
                                            999 + local_root,
                                            local_rank, local_root,
                                            group_index == sending_group )

    print( "        Rank #{:d}: {}".format( world_rank, value ) )

    intercomm.Barrier()

    if group_index == sending_group and local_rank == local_root:
        print( "\n" )

    intercomm.Barrier()

    # 2. second group broadcasts to first group.

    local_root    = 0
    sending_group = 1

    if group_index == sending_group and local_rank == local_root:
        print( ("   2. Second Group to First Group\n" +
                "\n" +
                "      All ranks in second group participate in the collective, though do not receive\n" +
                "      the data broadcast by the rank amongst them.  Only the ranks in the first\n" +
                "      group receive the data.\n" +
                "\n" ) )

    # synchronize all of the ranks so our banner is more likely to occur before
    # any output below.
    intercomm.Barrier()

    value = MPIUtility.intercomm_broadcast( intercomm,
                                            999 + local_root,
                                            local_rank, local_root,
                                            group_index == sending_group )

    print( "        Rank #{:d}: {}".format( world_rank, value ) )

    intercomm.Barrier()

    if group_index == sending_group and local_rank == local_root:
        print( "\n" )

    intercomm.Barrier()

    # 3. first group broadcasts to second group, though a non-zero local rank
    #    sends the data.
    #
    #    XXX: this assumes we have at least two ranks in the first group.

    local_root    = 1
    sending_group = 0

    if group_index == sending_group and local_rank == local_root:
        print( ("   1. First Group to Second Group with Non-zero Root Rank\n" +
                "\n" +
                "      All ranks in first group participate in the collective, though do not receive\n" +
                "      the data broadcast by the rank amongst them.  Only the ranks in the second\n" +
                "      group receive the data.  Rank #{:d} in the first group sends data.\n" +
                "\n" ).format( local_root) )

    # synchronize all of the ranks so our banner is more likely to occur before
    # any output below.
    intercomm.Barrier()

    value = MPIUtility.intercomm_broadcast( intercomm,
                                            999 + local_root,
                                            local_rank, local_root,
                                            group_index == sending_group )

    print( "        Rank #{:d}: {}".format( world_rank, value ) )

    intercomm.Barrier()

    if group_index == sending_group and local_rank == local_root:
        print( "\n" )

    intercomm.Barrier()

# identify who we are in and how big the global intracommunicator is.
world_comm = MPI.COMM_WORLD
world_rank = world_comm.Get_rank()
world_size = world_comm.Get_size()

# split the world into two pieces.
if world_rank  < (world_size // 2):
    group_index = 0
else:
    group_index = 1

# create an intracommunicator for the ranks in each half.
our_comm = world_comm.Split( group_index, key=world_rank )

# create an intercommunicator between the two halves.  rank 0 in the universe is
# the root in the first half, and rank (world_size/2) is the root in the second
# half.
if group_index == 0:
    inter_comm = our_comm.Create_intercomm( 0, world_comm, world_size // 2 )
else:
    inter_comm = our_comm.Create_intercomm( 0, world_comm, 0 )

# show how to broadcast data between the two halves.
demonstrate_broadcast( inter_comm, our_comm.Get_rank(), world_rank, group_index )
