from multiprocessing import Pool
import numpy as np

def multiprocess_task_on_list(task, list_input, n_jobs):
    """
        task: function that applies to each element of the list
        list_input: list of elements
        n_jobs: number of subprocesses to be used
    """
    proc_pool = Pool(n_jobs)
    list_output = proc_pool.map(task, list_input)
    proc_pool.close()
    return list_output

def multiprocess_task_many_args(task, zipped_input, n_jobs):
    """
        task: function that applies to each element of the list
        zipped_input: zip(arg1_list, arg2_list, ...)
        n_jobs: number of subprocesses to be used
    """
    proc_pool = Pool(n_jobs)
    list_output = proc_pool.starmap(task, zipped_input)
    proc_pool.close()
    return list_output
