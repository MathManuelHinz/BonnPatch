# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.stdlib   cimport rand, srand, RAND_MAX, malloc, realloc, free
from libc.math     cimport log
from libc.time     cimport time
cimport cython

ctypedef double DTYPE_t

cdef inline double uniform() nogil:
    """
    Return a double in [0,1) using rand().
    """
    return rand() / (RAND_MAX + 1.0)

cdef inline DTYPE_t expovariate(double scale) nogil:
    """
    Sample an exponential random variable with given scale.
    """
    return -scale * log(uniform())

cpdef tuple sample_path(object Y, double T=100.0, double t0=0.0):
    """
    Cython version of sample_path using nogil and the C stdlib.
    Returns two Python lists: arrival times and visited states.
    """
    cdef:
        # Python‐level fetch under GIL
        double[:, :] qview
        double[:]     piview
        int           N
        int           s, i
        double        t, scale, tau
        double        r, cum
        # C arrays for times and states
        DTYPE_t      *ts_arr
        int          *xs_arr
        Py_ssize_t    size, capacity

    # Acquire GIL to get generator matrix and initial distribution
    q_py, pi_py = Y.get_generator_initial()
    qview = q_py    # memoryview over 2D C‐contiguous array
    piview = pi_py  # memoryview over 1D C‐contiguous array

    N = qview.shape[0]
    srand(<unsigned int>time(NULL))  # seed RNG

    # Initial allocation
    capacity = 16
    ts_arr   = <DTYPE_t *>malloc(capacity * sizeof(DTYPE_t))
    xs_arr   = <int     *>malloc(capacity * sizeof(int))
    size     = 0

    # Start at t0
    t  = t0
    # Sample initial state from pi
    r  = uniform()
    cum = 0.0
    for i in range(N):
        cum += piview[i]
        if r < cum:
            s = i
            break

    # Record initial event
    ts_arr[size] = t
    xs_arr[size] = s
    size += 1

    # Main loop: sample until time T
    with nogil:
        while t < T:
            scale = -1.0 / qview[s, s]
            tau   = expovariate(scale)
            t    += tau
            if t >= T:
                break

            # Sample next state based on outgoing rates
            r   = uniform()
            cum = 0.0
            for i in range(N):
                if i == s:
                    continue
                cum += qview[s, i] * scale
                if r < cum:
                    s = i
                    break

            # Grow arrays if needed
            if size == capacity:
                capacity <<= 1
                ts_arr = <DTYPE_t *>realloc(ts_arr, capacity * sizeof(DTYPE_t))
                xs_arr = <int     *>realloc(xs_arr, capacity * sizeof(int))

            ts_arr[size] = t
            xs_arr[size] = s
            size += 1

        # Append final time T and hold state
        ts_arr[size] = T
        xs_arr[size] = xs_arr[size - 1]
        size += 1

    # Convert to Python lists and clean up

    py_ts = [0.0] * size
    py_xs = [0]   * size
    for i in range(size):
        py_ts[i] = ts_arr[i]
        py_xs[i] = xs_arr[i]

    free(ts_arr)
    free(xs_arr)

    return py_ts, py_xs
