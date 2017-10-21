#!/usr/bin/env python

from mpi4py import MPI

def partition_by_string( intracomm, color_string ):
    # XXX: adapted from:
    #
    # https://stackoverflow.com/questions/35924226/openmpi-mpmd-get-communication-size
    #

    # identify the longest color string so we can size our buffer properly.
    #
    # NOTE: since we're using a Pythonic approach to MPI we don't actually need
    #       this value.  we'd use it to allocate a buffer of
    #       intracomm_size * maximum_color_length to call MPI_Allgather() into.
    #maximum_name_length = intracomm.allreduce( len( color_string ), max )

    # collect the names of every application in our universe.
    colors = intracomm.allgather( color_string )

    # loop through concatenated names to find the first one that matches,
    # track that index to split the groups by.
    color_index = colors.index( color_string )

    return intracomm.Split( color_index, key=intracomm.Get_rank() )

def intercomm_broadcast( intercomm, send_value, local_rank, local_root, sending_flag ):
    """
    Wrapper for broadcasting a value over an intercommunicator.  Since
    broadcasting from one group to another via an intercommunicator is a
    superset of a broadcast across a single group via an intracommunicator,
    this function handles setting the broadcast root within each group.

    Takes 5 arguments:

      intercomm     - Intercommunicator used to broadcast data from one
                      group to another.
      send_value    - Broadcast value.  This is only used on the root
                      rank.
      local_rank    - Rank in the local group.
      local_root    - Rank in the local, sending group who performs the
                      broadcast.
      sending_flag  - Flag indicating whether this rank is in the sending
                      or receiving group.

    Returns 1 value:

      value - Value received from the broadcast operation.  For the
              receiving group, this will be send_value and None for everyone
              else.

     """

    if sending_flag:
        # the sending group needs to identify the sender (root) and the
        # passive ranks.
        if local_rank == local_root:
            # declare ourself the root.
            value      = send_value
            local_root = MPI.ROOT
        else:
            # declare ourself NOT the root.
            value      = None
            local_root = MPI.PROC_NULL
    else:
        # the receiving group identifies the root in the sending group.
        # note that we use the local rank for the sender.
        value = None

    return intercomm.bcast( value, root=local_root )
