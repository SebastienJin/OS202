# TD n°3 - parallélisation du Bucket Sort

\* Ce TD peut être réalisé au choix, en C++ ou en Python

Implémenter l'algorithme "bucket sort" tel que décrit sur les deux dernières planches du cours n°3 :

- le process 0 génère un tableau de nombres arbitraires,
- il les dispatch aux autres process,
- tous les process participent au tri en parallèle,
- le tableau trié est rassemblé sur le process 0.

# Explanation

When running the program with the command "mpiexec -np N python para_bucket_sort.py", 
where N is the number of processes, 
process 0 will generate a random set of data equal to 10 times the number of remaining processes. 
Then, each process will receive 10 pieces of data, perform a local sort, 
and then send the data back to process 0 to obtain the final sorted result.
