"""
    common functions

"""

# time lib
from time import time
# types
from typing import List
## nn libs
from torch import Tensor


# data types
Points = List[Tensor]  # size == [n,2]
Guards = Tensor # size == [,] * sol_count



def timer_func(func):
    """
    This function shows the execution time of the function object passed

    """
    def wrap_func(*args, **kwargs):
        t_1 = time()
        result = func(*args, **kwargs)
        t_2 = time()
        print(f'Function {args[0].__class__.__name__}.{func.__name__} executed in {(t_2-t_1):.4f}s')
        return result
    return wrap_func



def conditional_decorator(dec, condition):
    """
        conditional decorator for timing function
    """
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)
    return decorator
