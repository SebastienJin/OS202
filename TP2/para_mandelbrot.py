from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
import matplotlib.cm as cm
import matplotlib.cm
from typing import Union
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> Union[int, float]:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


def compute_rows(start_row, end_row, width, mandelbrot_set):
    rows = end_row - start_row
    convergence = np.empty((rows, width), dtype=np.double)
    scaleX = 3./width
    scaleY = 2.25/(end_row - start_row)
    for y in range(rows):
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY*(start_row+y))
            convergence[x, y] = mandelbrot_set.convergence(c, smooth=True)
    return convergence


mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024
rows_per_rank = height // size
extra_rows = height % size

if rank == 0:

    # Calcul de l'ensemble de mandelbrot :
    deb = time()

    start_row = rows_per_rank + extra_rows
    for i in range(1, size):
        comm.send((start_row, start_row + rows_per_rank), dest=i)
        start_row += rows_per_rank

    convergence = compute_rows(0, rows_per_rank + extra_rows, width, mandelbrot_set)
    for i in range(1, size):
        start_row, end_row = comm.recv(source=i)
        recv_data = comm.recv(source=i)
        convergence = np.vstack((convergence, recv_data))   
 
    fin = time()
    print(f"Rank0: Temps du calcul de l'ensemble de Mandelbrot : {fin-deb}")
    
    # Constitution de l'image résultante :
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
    fin = time()
    print(f"Rank0: Temps de constitution de l'image : {fin-deb}")
    image.show()
else:
    start_row, end_row = comm.recv(source=0)
    convergence = compute_rows(start_row, end_row, width, mandelbrot_set)
    comm.send((start_row, end_row), dest=0)
    comm.send(convergence, dest=0)
