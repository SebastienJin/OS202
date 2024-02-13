from mpi4py import MPI
import numpy as np

def bucket_sort(data, num_buckets=10):
    max_value = np.max(data)
    min_value = np.min(data)
    interval = (max_value - min_value) / num_buckets
    
    buckets = [[] for _ in range(num_buckets)]
    for number in data:
        index = int((number - min_value) // interval)
        if index == num_buckets:
            index -= 1
        buckets[index].append(number)
    
    sorted_data = []
    for bucket in buckets:
        sorted_data.extend(sorted(bucket))
    
    return sorted_data

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    elements_per_process = 10

    data = None
    if rank == 0:
        data = np.random.randint(0, 100, size=size*elements_per_process).astype('i')
        print("Original data on process 0:")
        print(data)

    local_data = np.empty(elements_per_process, dtype='i')
    comm.Scatter(data, local_data, root=0)

    local_sorted = bucket_sort(local_data, num_buckets=5)

    sorted_data = None
    if rank == 0:
        sorted_data = np.empty_like(data)
    comm.Gather(np.array(local_sorted, dtype='i'), sorted_data, root=0)

    if rank == 0:
        final_sorted = bucket_sort(sorted_data, num_buckets=size)
        print("Sorted data on process 0:")
        print(final_sorted)

if __name__ == "__main__":
    main()
