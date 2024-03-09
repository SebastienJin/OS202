from mpi4py import MPI
import sys
import time
import numpy as np
import pygame as pg
import maze
import pheromone
import ants
from ants import Colony, display
from maze import maze_display
from functools import reduce

DEBUG = False

if __name__ == "__main__":
    
    # MPI init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Window init and hyper params
    deb = time.time()
    pg.init()
    size_laby = 25, 25
    if len(sys.argv) > 2:
        size_laby = int(sys.argv[1]), int(sys.argv[2])

    resolution = size_laby[1]*8, size_laby[0]*8
    if rank == 0:
        screen = pg.display.set_mode(resolution)
    else:
        screen = None

    max_life = 500
    if len(sys.argv) > 3:
        max_life = int(sys.argv[3])

    pos_food = size_laby[0]-1, size_laby[1]-1
    pos_nest = 0, 0
        
    a_maze = maze.Maze(size_laby, 12345)
    if rank == 0:
        mazeImg = maze_display(a_maze)
        screen.blit(mazeImg, (0, 0))
        pg.display.update()

    alpha = 0.9
    beta  = 0.99
    if len(sys.argv) > 4:
        alpha = float(sys.argv[4])
    if len(sys.argv) > 5:
        beta = float(sys.argv[5])

    # Ants distribution to slave procs
    if DEBUG:
        nb_ants = 9
    else:
        nb_ants = size_laby[0]*size_laby[1]//4
    ants_per_proc = nb_ants // (size - 1)
    start_index = (rank - 1) * ants_per_proc
    end_index = rank * ants_per_proc if rank != size - 1 else nb_ants
    ants_per_proc_local = end_index - start_index
    if DEBUG and rank == 0:
        print(
            f"\n*** RUNNING PARALLEL ACO WITH PARAMS: ***\n"
            f"\tRESOLUTION = {resolution[0]}x{resolution[1]}\n"
            f"\tnb_ants = {nb_ants} "
            f"ants_per_proc = {ants_per_proc}\n"
            f"*****************************************")
        pg.time.wait(1000)

    # Ants init
    # ants = Colony(nb_ants, pos_nest, max_life, start_index, end_index)
    snapshot_taken = False
    pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
    if rank != 0:
        ants = Colony(start_index, end_index, pos_nest, max_life)
        local_food_counter = np.array([0])
    if rank == 0:
        updated_pherom = None
        total_counter = np.array([0])
    if DEBUG: print("ANTS INIT DONE.")

    # while True:
    for loop in range(3000):
        if rank == 0:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit(0)

        if rank == 0:
            # gathered_colony, local_food_counter, local_pherom = comm.recv(source=1)
            # total_counter += local_food_counter
            # if size > 2:
            #     for i in range(2, size):
            #         colony, local_food_counter, local_pherom = comm.recv(source=i)
            #         gathered_colony.merge(colony)
            #         total_counter += local_food_counter
            #         # pherom
            # pherom = local_pherom #TODO
            # total_colony = gathered_colony

            # Receive data from slave procs and aggregate to global variable
            all_received_data = [comm.recv(source=i) for i in range(1, size)]
            if DEBUG:
                for i in range(len(all_received_data)):
                    print(f"rk{rank} GOT: 0:{all_received_data[i][0]}, 1:{all_received_data[i][1]}, 2:{all_received_data[i][2]}")

            all_local_colony = [item[0] for item in all_received_data]
            total_colony = reduce(lambda col1, col2: col1.merge(col2), all_local_colony)

            all_local_counter = [item[1] for item in all_received_data]
            total_counter = reduce(lambda x, y: x + y, all_local_counter)

            all_local_pherom_matrix = [item[2].pheromon for item in all_received_data]
            updated_pherom = np.mean(np.array(all_local_pherom_matrix), axis=0)
            # pherom.update(updated_pherom)
            if DEBUG: print("DATA GATHERED")

            # When snapshot not taken, proc send a message with local snapshot taken flag, so it's valid to unzip a 4th elem
            # In this case consider potential update to snapshot taken flag, return this message to slaves
            if not snapshot_taken:
                # all_local_snapshot_taken = [item[3] for item in all_received_data]
                # snapshot_taken = any(all_local_snapshot_taken)
                snapshot_taken = np.any(np.array(all_local_counter) != 0)
                message = [pherom, snapshot_taken]
                
                if snapshot_taken: # First food print during this master aggregation
                    if DEBUG: pg.time.wait(1000)
                    i_proc_first_food, _ = np.nonzero(np.array(all_local_counter))
                    if DEBUG: print(f"np.nonzero(np.array(all_local_counter))={np.nonzero(np.array(all_local_counter))}")
                    i_proc_first_food = i_proc_first_food[0] # np.array
                    if DEBUG: print(f"rk {i_proc_first_food} got the first food")
                    pherom.update(all_local_pherom_matrix[i_proc_first_food])
                    pherom.display(screen)
                    screen.blit(mazeImg, (0, 0))
                    display(total_colony, screen)
                    pg.display.update()
                    pg.image.save(screen, "MyFirstFood.png")
                    if DEBUG: pg.time.wait(1000)
            else:
                message = [pherom]
            # TODO A potential bug here is when strategy changes, can be there >1 procs getting global first snapshot.
                
            pherom.update(updated_pherom)

            # event_list = pg.event.get()
            # if len(event_list) > 0:
            #     print(event_list)
            pherom.display(screen)
            screen.blit(mazeImg, (0, 0))
            display(total_colony, screen) # ants.display
            pg.display.update()

            for i in range(1, size):
                comm.send(message, dest=i)

            
            # print(f"FPS : {1./(end-deb):6.2f}, nourriture : {food_counter:7d}", end='\r')
        else:
            # Update and display pheromones, ants
            local_food_counter[0] = ants.advance(a_maze, pos_food, pos_nest, pherom, local_food_counter[0])
            if local_food_counter == 1 and not snapshot_taken:
                if DEBUG: pg.time.wait(2000)
                # snapshot_taken = True
                print(f"Rank {rank} got our first food!")
                # pg.image.save(screen, "MyFirstFood.png")

            pherom.do_evaporation(pos_food)
            message = [ants, local_food_counter, pherom]
            # if not snapshot_taken:
            #     message.append(snapshot_taken)
            comm.send(message, dest=0)
            if DEBUG: print(f"rk{rank} returning a message of length {len(message)} with local snap flag is {snapshot_taken}")

            message = comm.recv(source=0)
            pherom = message[0]
            if len(message) > 1: snapshot_taken = message[1]
            if DEBUG: print(f"type(pheromon) after message from master = {type(pherom)}")
    end = time.time()
    print(f"Time taken: {end-deb}")



"""
Several pheromon-aggregate strategy:
 1 - Each iter when slave update (even evaporation), it will force master to align it to all procs
 2 - when one slave update its pheromon due to finding food and return it to master, master proc will immediately broadcast it to all slaves
 3 - master proc will wait until all slaves return its pheromon update from finding food, and aggregate, then broadcast the overall update

This code serve for the ? functionality
"""
