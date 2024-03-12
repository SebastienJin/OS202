import numpy as np
import sys
import functools
import operator
import pygame  as pg

class Grille:
    """
    Grille torique décrivant l'automate cellulaire.
    En entrée lors de la création de la grille :
        - dimensions est un tuple contenant le nombre de cellules dans les deux directions (nombre lignes, nombre colonnes)
        - init_pattern est une liste de cellules initialement vivantes sur cette grille (les autres sont considérées comme mortes)
        - color_life est la couleur dans laquelle on affiche une cellule vivante
        - color_dead est la couleur dans laquelle on affiche une cellule morte
    Si aucun pattern n'est donné, on tire au hasard quels sont les cellules vivantes et les cellules mortes
    Exemple :
       grid = Grille( (10,10), init_pattern=[(2,2),(0,2),(4,2),(2,0),(2,4)], color_life=pg.Color("red"), color_dead=pg.Color("black"))
    """
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        import random
        self.dimensions = dim
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            indices_i = [v[0] for v in init_pattern]
            indices_j = [v[1] for v in init_pattern]
            self.cells[indices_i,indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self):
        """
        Calcule la prochaine génération de cellules en suivant les règles du jeu de la vie
        """
        # Remarque 1: on pourrait optimiser en faisant du vectoriel, mais pour plus de clarté, on utilise les boucles
        # Remarque 2: on voit la grille plus comme une matrice qu'une grille géométrique. L'indice (0,0) est donc en haut
        #             à gauche de la grille !
        ny = self.dimensions[0]
        nx = self.dimensions[1]
        next_cells = np.empty(self.dimensions, dtype=np.uint8)
        diff_cells = []
        for i in range(ny):
            i_above = (i+ny-1)%ny
            i_below = (i+1)%ny
            for j in range(nx):
                j_left = (j-1+nx)%nx
                j_right= (j+1)%nx
                voisins_i = [i_above,i_above,i_above, i     , i      , i_below, i_below, i_below]
                voisins_j = [j_left ,j      ,j_right, j_left, j_right, j_left , j      , j_right]
                voisines = np.array(self.cells[voisins_i,voisins_j])
                nb_voisines_vivantes = np.sum(voisines)
                if self.cells[i,j] == 1: # Si la cellule est vivante
                    if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                        next_cells[i,j] = 0 # Cas de sous ou sur population, la cellule meurt
                        diff_cells.append(i*nx+j)
                    else:
                        next_cells[i,j] = 1 # Sinon elle reste vivante
                elif nb_voisines_vivantes == 3: # Cas où cellule morte mais entourée exactement de trois vivantes
                    next_cells[i,j] = 1         # Naissance de la cellule
                    diff_cells.append(i*nx+j)
                else:
                    next_cells[i,j] = 0         # Morte, elle reste morte.
        self.cells = next_cells
        return diff_cells


class App:
    """
    Cette classe décrit la fenêtre affichant la grille à l'écran
        - geometry est un tuple de deux entiers donnant le nombre de pixels verticaux et horizontaux (dans cet ordre)
        - grid est la grille décrivant l'automate cellulaire (voir plus haut)
    """
    def __init__(self, geometry, grid):
        self.grid = grid
        # Calcul de la taille d'une cellule par rapport à la taille de la fenêtre et de la grille à afficher :
        self.size_x = geometry[1]//grid.dimensions[1]
        self.size_y = geometry[0]//grid.dimensions[0]
        if self.size_x > 4 and self.size_y > 4 :
            self.draw_color=pg.Color('lightgrey')
        else:
            self.draw_color=None
        # Ajustement de la taille de la fenêtre pour bien fitter la dimension de la grille
        self.width = grid.dimensions[1] * self.size_x
        self.height= grid.dimensions[0] * self.size_y
        # Création de la fenêtre à l'aide de tkinter
        self.screen = pg.display.set_mode((self.width,self.height))
        #
        self.canvas_cells = []

    def compute_rectangle(self, i: int, j: int):
        """
        Calcul la géométrie du rectangle correspondant à la cellule (i,j)
        """
        return (self.size_x*j, self.height - self.size_y*i - 1, self.size_x, self.size_y)

    def compute_color(self, i: int, j: int):
        if self.grid.cells[i,j] == 0:
            return self.grid.col_dead
        else:
            return self.grid.col_life

    def draw(self):
        [self.screen.fill(self.compute_color(i,j),self.compute_rectangle(i,j)) for i in range(self.grid.dimensions[0]) for j in range(self.grid.dimensions[1])]
        if (self.draw_color is not None):
            [pg.draw.line(self.screen, self.draw_color, (0,i*self.size_y), (self.width,i*self.size_y)) for i in range(self.grid.dimensions[0])]
            [pg.draw.line(self.screen, self.draw_color, (j*self.size_x,0), (j*self.size_x,self.height)) for j in range(self.grid.dimensions[1])]
        pg.display.update()

dico_patterns = { # Dimension et pattern dans un tuple
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((200,100),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }


if len(sys.argv)>1:
    choice = sys.argv[1]
else:
    choice='acorn'
dim = dico_patterns[choice][0]

# Importation du module MPI
from mpi4py import MPI
comme_global = MPI.COMM_WORLD.Dup()
rank_global = comme_global.rank
size_global = comme_global.size

global_grille=None

if size_global == 1:
    init_pattern = dico_patterns[choice]
    global_grille=Grille(*init_pattern)
    
    global_grid = global_grille.cells
    appli = App((800,800),global_grille)
    while True:
       diff = global_grille.compute_next_iteration()
       appli.draw()

elif size_global == 2:
    # séparation entre le processus qui affiche et les processus qui calculent
    if rank_global == 0:
        init_pattern = dico_patterns[choice]
        grid = Grille(*init_pattern)
        appli = App((800, 800), grid)
        while True:
            comme_global.send(grid, dest=1, tag=120)
            cells = comme_global.recv(source=1, tag=130)
            grid.cells = cells
            appli.draw()
    else:
        while True: 
            grid = comme_global.recv(source=0, tag=120)
            grid.compute_next_iteration()
            comme_global.send(grid.cells, dest=0, tag=130)

else:
    if rank_global == 0:
        status = 0
        
        init_pattern = dico_patterns[choice]
        global_grille=Grille(*init_pattern)
        
        global_grid = global_grille.cells
        appli = App((800,800),global_grille)
    else :
        status = 1

    # On divise les processus en deux groupes
    comm = comme_global.Split(status,rank_global)
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank_global==0:
        comme_global.send(global_grille,dest=1,tag=0)
    elif rank_global==1:
        global_grille=comme_global.recv(source=0,tag=0)

    if rank_global==0:
        comme_global.send(global_grille,dest=1,tag=0)
    elif rank_global==1:
        global_grille=comme_global.recv(source=0,tag=0)

    while True:
        if status==1:
            sendbuff=None
            recvbuf=None
            residu_a_fin=False
            if dim[0]%size==0:
                rows_per_process=int(dim[0]/size)
            else:
                residu_a_fin=True
                rows_per_process=dim[0]//size

            '''
            On utilise la fonction Scatterv pour envoyer les données à chaque processus. 
            Pour cela, on a besoin de définir les paramètres sendcounts et displacements.
            '''    
            nom_elements_par_processus=rows_per_process*dim[1]

            sendcounts = [rows_per_process * dim[1] for _ in range(size)]

            extra_rows = dim[0] % size

            sendcounts[-1] += extra_rows * dim[1]

            displacements = [sum(sendcounts[:i]) for i in range(size)]

            recvbuf_size = sendcounts[rank]
            recvbuf = np.empty(recvbuf_size, dtype=np.uint8)

            if rank==0:
                sendbuff=global_grille.cells
                sendbuff=sendbuff.flatten()
                assert sum(sendcounts) == len(sendbuff), "Total sendcounts does not match sendbuff size."

            try:
                comm.Scatterv([sendbuff, sendcounts, displacements, MPI.UNSIGNED_CHAR], recvbuf, root=0)
            except Exception as e:
                if rank == 0:
                    print(f"Error in comm.Scatterv at rank {rank}: {e}")
                    print("sendcounts:", sendcounts)
                    print("displacements:", displacements)
                    print("sendbuff size:", len(sendbuff))
                exit()
                
            # On récupère les vecteurs up et down    
            vector_up=np.empty((1,dim[1]),dtype=np.uint8)
            vector_down=np.empty((1,dim[1]),dtype=np.uint8)
            vecteu_up_r=None
            vecteu_down_r=None

            if residu_a_fin and rank==size-1:
                recvbuf=recvbuf.reshape((int(sendcounts[-1]/dim[1]),dim[1]))
            else:
                recvbuf=recvbuf.reshape((rows_per_process,dim[1]))
                
            if rank==0:
                vector_up=recvbuf[0]
                vector_down=recvbuf[recvbuf.shape[0]-1]
                vecteu_up_r=np.empty(vector_up.shape,dtype=np.uint8)
                vecteu_down_r=np.empty(vector_down.shape,dtype=np.uint8)
                comm.Send(vector_up,dest=size-1,tag=0)
                comm.Send(vector_down,dest=1,tag=1)
                comm.Recv(vecteu_up_r,source=size-1,tag=2*size-1)
                comm.Recv(vecteu_down_r,source=1,tag=2)
            elif rank==size-1:
                vector_up=recvbuf[0]
                vector_down=recvbuf[recvbuf.shape[0]-1]
                vecteu_up_r=np.empty(vector_up.shape,dtype=np.uint8)
                vecteu_down_r=np.empty(vector_down.shape,dtype=np.uint8)
                comm.Send(vector_up,dest=rank-1,tag=2*rank)
                comm.Send(vector_down,dest=0,tag=2*rank+1)
                comm.Recv(vecteu_up_r,source=rank-1,tag=2*(rank-1)+1)
                comm.Recv(vecteu_down_r,source=0,tag=0)
            else:
                vector_up=recvbuf[0]
                vector_down=recvbuf[recvbuf.shape[0]-1]
                vecteu_up_r=np.empty(vector_up.shape,dtype=np.uint8)
                vecteu_down_r=np.empty(vector_down.shape,dtype=np.uint8)
                comm.Send(vector_up,dest=rank-1,tag=2*rank)
                comm.Send(vector_down,dest=rank+1,tag=2*rank+1)
                comm.Recv(vecteu_up_r,source=rank-1,tag=2*(rank-1)+1)
                comm.Recv(vecteu_down_r,source=rank+1,tag=2*rank+2)

            next_cells=recvbuf.copy()
            
            recvbuf=np.vstack((vecteu_up_r,recvbuf))
            recvbuf=np.vstack((recvbuf,vecteu_down_r))
            
            diff_loc=[]
            def coordinates_gl(i,j):
                return nom_elements_par_processus*rank+i*dim[1]+j+1
            ny = recvbuf.shape[0]
            nx = recvbuf.shape[1]

            for i in range(1,rows_per_process+1):
                i_above = i-1
                i_below = i+1 
                for j in range(dim[1]):
                    j_left = (j-1+nx)%nx
                    j_right= (j+1)%nx
                    voisins_i = [i_above,i_above,i_above, i     , i      , i_below, i_below, i_below]
                    voisins_j = [j_left ,j      ,j_right, j_left, j_right, j_left , j      , j_right]
                    voisines = np.array(recvbuf[voisins_i,voisins_j])
                    nb_voisines_vivantes = np.sum(voisines)
                    if recvbuf[i,j] == 1: # Si la cellule est vivante
                        if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                            next_cells[i-1,j-1] = 0 # Cas de sous ou sur population, la cellule meurt
                            ig=coordinates_gl(i-1,j-1)
                            diff_loc.append(ig)
                        else:
                            next_cells[i-1,j-1] = 1 # Sinon elle reste vivante
                    elif nb_voisines_vivantes == 3: # Cas où cellule morte mais entourée exactement de trois vivantes
                        next_cells[i-1,j-1] = 1         # Naissance de la cellule
                        ig=coordinates_gl(i-1,j-1)
                        diff_loc.append(ig)
                    else:
                        next_cells[i-1,j-1] = 0
        
            diff_glob=comm.gather(diff_loc,root=0)
            if rank==0:
                diff_glob=functools.reduce(operator.iconcat,diff_glob,[])
                comme_global.send(diff_glob,dest=0,tag=13000)

        #On recoit les différences et les applique
        if status==0:
            diff_glob=comme_global.recv(source=1,tag=13000)
            global_grid=global_grille.cells.copy()
            glob_grid_shape=global_grid.shape
            global_grid=global_grid.flatten()
            for ind in diff_glob:
                if global_grid[ind]==1:
                    global_grid[ind]=0
                else:
                    global_grid[ind]=1
            global_grille.cells=global_grid.reshape(glob_grid_shape)

            comme_global.send(global_grille,dest=1,tag=150)
            appli.draw()

        if rank_global==1:
            global_grille=comme_global.recv(source=0,tag=150)