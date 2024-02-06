from mpi4py import MPI
import numpy as np

# Initialize the MPI environment
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Dimension of the problem (can be changed)
dim = 120

# Assuming dim is divisible by the number of processes
n_loc = dim // size  

# Initialize the matrix A and vector u
A_full = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
u = np.array([i+1. for i in range(dim)])

if rank == 0:
    print(f"A_full = {A_full}")
    print(f"u = {u}")

# Split by columns
A = np.array_split(A_full, size, axis=1)[rank]  

v_partial = np.dot(A, u)

v = np.zeros(dim) if rank == 0 else None

comm.Gather(v_partial, v, root=0)

if rank == 0:
    print(f"v = {v}")

