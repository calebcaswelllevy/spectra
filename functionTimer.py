import time
def measure_time(func):
    """
    A decorator to measure execution time of an arbitrary function

    func: a function to measure
    """

    def wrapper(*args, **kargs):
        t = time.time()
        result = func(*args, **kargs)
        print(f"{func.__name__} took {t} seconds")
        return result
    return wrapper

