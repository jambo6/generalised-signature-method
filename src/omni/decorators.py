import time


def timeit(method):
    """ Get the time it takes for a method to run.

    Args:
        method (function): The function to time.

    Returns:
        Method wrapped with an operation to time it.
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r \n  %2.2f ms' % (method, (te - ts) * 1000))
        return result
    return timed
