# cython: boundscheck=False, wraparound=True, cdivision=True
# cython: language_level=3, infer_types=True
from libc.stdlib cimport calloc, malloc, realloc, free
from libc.string cimport memset
from cython.parallel import prange, parallel
cimport cython

ctypedef double DTYPE_t

cdef Py_ssize_t MAX_EVENTS = 10000000

@cython.inline
cdef inline DTYPE_t max0(DTYPE_t x) nogil:
    return x if x > 0.0 else 0.0

@cython.inline  
cdef inline DTYPE_t min0(DTYPE_t x) nogil:
    return x if x < 0.0 else 0.0

@cython.inline
cdef inline DTYPE_t sync_cutting(DTYPE_t x, DTYPE_t g) nogil:
    return x * (g >= 0.0)

cpdef tuple higher_order_hinkley_detector_display(double[:] z_view,
                                                 double mu_0 = 0.0,
                                                 double mu_1 = 1.0,
                                                 double cutoff = 2.0,
                                                 int order = 8):
    """
    Cython version of higher_order_hinkley_detector_display using only
    the C stdlib and Python built-ins.
    """
    cdef:
        Py_ssize_t MAX_EVENTS = 10000000
        Py_ssize_t T = z_view.shape[0]
        DTYPE_t **gs       = <DTYPE_t**> malloc(order * sizeof(DTYPE_t*))
        Py_ssize_t i, t, j
        DTYPE_t mu_bar     = 0.5 * (mu_0 + mu_1)
        DTYPE_t p          = 0.5 * (mu_1 - mu_0)
        bint is_lower_state
        int *jump_idxs_c   = <int*> malloc((T + 1) * sizeof(int))
        int *jump_vals_c   = <int*> malloc((T + 1) * sizeof(int))
        Py_ssize_t jumps   = 0
        list return_gs     = []
        list jump_idxs_py  = []
        list jump_vals_py  = []
        list filtered_z    = []
        list segment_lengths
        DTYPE_t *seg_arr
        Py_ssize_t seg_len

    # Allocate each gs[i] array of length T
    for i in range(order):
        gs[i] = <DTYPE_t*> malloc(T * sizeof(DTYPE_t))
        for j in range(T):
            gs[i][j] = 0.0

    # Initial state
    is_lower_state = (z_view[0] - mu_bar) <= 0
    jump_idxs_c[0]  = 0
    jump_vals_c[0]  = (z_view[0] - mu_bar) > 0
    jumps = 1

    # Main loop
    t = 1
    while t < T:
        if is_lower_state:
            # Level 0 detection
            gs[0][t] = (gs[0][t-1] + z_view[t] - mu_bar)
            if gs[0][t] < 0:
                gs[0][t] = 0.0
            # Higher-order detectors
            for i in range(1, order):
                gs[i][t] = sync_cutting(gs[i][t-1] + gs[i-1][t], gs[0][t])
            # Check threshold crossing
            if gs[order-1][t] >= cutoff:
                # Copy segment [0..t] scaled by p/cutoff
                seg_len = t + 1
                seg_arr = <DTYPE_t*> malloc(seg_len * sizeof(DTYPE_t))
                for j in range(seg_len):
                    seg_arr[j] = (p * gs[order-1][j]) / cutoff
                # Convert to Python list and append
                return_gs.append([seg_arr[k] for k in range(seg_len)])
                free(seg_arr)
                # Find last zero crossing
                j = t
                while j >= 0 and gs[order-1][j] != 0.0:
                    j -= 1
                t = j
                is_lower_state = False
                jump_idxs_c[jumps] = t + 1
                jump_vals_c[jumps] = 1
                jumps += 1
                # Reset intermediate gs at index j
                for i in range(order-1):
                    gs[i][j] = 0.0
        else:
            # Upper-state detection
            gs[0][t] = (gs[0][t-1] + z_view[t] - mu_bar)
            if gs[0][t] > 0:
                gs[0][t] = 0.0
            for i in range(1, order):
                gs[i][t] = sync_cutting(gs[i][t-1] + gs[i-1][t], -gs[0][t])
            if gs[order-1][t] <= -cutoff:
                seg_len = t + 1
                seg_arr = <DTYPE_t*> malloc(seg_len * sizeof(DTYPE_t))
                for j in range(seg_len):
                    seg_arr[j] = (p * gs[order-1][j]) / cutoff
                return_gs.append([seg_arr[k] for k in range(seg_len)])
                free(seg_arr)
                j = t
                while j >= 0 and gs[order-1][j] != 0.0:
                    j -= 1
                t = j
                is_lower_state = True
                jump_idxs_c[jumps] = t + 1
                jump_vals_c[jumps] = 0
                jumps += 1
                for i in range(order-1):
                    gs[i][j] = 0.0
        t += 1

    # Final segment up to T
    seg_len = T
    seg_arr = <DTYPE_t*> malloc(seg_len * sizeof(DTYPE_t))
    for j in range(seg_len):
        seg_arr[j] = (p * gs[order-1][j]) / cutoff
    return_gs.append([seg_arr[k] for k in range(seg_len)])
    free(seg_arr)

    # Build Python jump lists
    for i in range(jumps):
        jump_idxs_py.append(jump_idxs_c[i])
        jump_vals_py.append(jump_vals_c[i])

    free(jump_idxs_c)
    free(jump_vals_c)

    # Compute segment lengths and filtered_z
    segment_lengths = []
    for i in range(jumps - 1):
        segment_lengths.append(jump_idxs_py[i+1] - jump_idxs_py[i])
    segment_lengths.append(T - jump_idxs_py[-1])
    for idx, length in enumerate(segment_lengths):
        filtered_z.extend([jump_vals_py[idx]] * int(length))

    # Free gs arrays
    for i in range(order):
        free(gs[i])
    free(gs)

    return return_gs, jump_idxs_py, jump_vals_py, filtered_z

cpdef list higher_order_hinkley_detector(double[:] z_view,
                                         double mu_0=0.0,
                                         double mu_1=1.0,
                                         double cutoff=2.0,
                                         int order=8):
    cdef:
        Py_ssize_t T = z_view.shape[0]
        Py_ssize_t t = 1, jumps = 0
        double mu_bar = 0.5 * (mu_0 + mu_1)
        bint is_lower = (z_view[0] - mu_bar) <= 0.0
        int *jump_idxs = <int*> malloc(MAX_EVENTS * sizeof(int))
        int *jump_vals = <int*> malloc(MAX_EVENTS * sizeof(int))
        int *segment_lengths = <int*> malloc(MAX_EVENTS * sizeof(int))
        double **gs = <double**> malloc(order * sizeof(double*))
        Py_ssize_t i, j, idx
        list filtered_z

    # Preallocate state arrays
    for i in range(order):
        gs[i] = <double*> calloc(T, sizeof(double))

    # Initialize first jump
    jump_idxs[0] = 0
    jump_vals[0] = (z_view[0] - mu_bar) > 0.0
    jumps = 1

    # Main detection loop
    with nogil:
        while t < T:
            if is_lower:
                gs[0][t] = max0(gs[0][t-1] + z_view[t] - mu_bar)
                for i in range(1, order):
                    gs[i][t] = sync_cutting(gs[i][t-1] + gs[i-1][t], gs[0][t])
                if gs[order-1][t] > cutoff:
                    # backtrack to last zero
                    j = t
                    while j >= 0 and gs[order-1][j] != 0.0:
                        j -= 1
                    t = j
                    is_lower = False
                    jump_idxs[jumps] = t + 1
                    jump_vals[jumps] = 1
                    jumps += 1
                    for i in range(order-1):
                        gs[i][j] = 0.0
            else:
                gs[0][t] = min0(gs[0][t-1] + z_view[t] - mu_bar)
                for i in range(1, order):
                    gs[i][t] = sync_cutting(gs[i][t-1] + gs[i-1][t], -gs[0][t])
                if gs[order-1][t] <= -cutoff:
                    j = t
                    while j >= 0 and gs[order-1][j] != 0.0:
                        j -= 1
                    t = j
                    is_lower = True
                    jump_idxs[jumps] = t + 1
                    jump_vals[jumps] = 0
                    jumps += 1
                    for i in range(order-1):
                        gs[i][j] = 0.0
            t += 1

    # Compute segment lengths
    for idx in range(jumps-1):
        segment_lengths[idx] = jump_idxs[idx+1] - jump_idxs[idx]
    segment_lengths[jumps-1] = T - jump_idxs[jumps-1]

    # Build filtered_z with preallocation
    cdef Py_ssize_t total_len = 0
    for idx in range(jumps):
        total_len += segment_lengths[idx]
    filtered_z = [0] * total_len

    cdef Py_ssize_t pos = 0
    for idx in range(jumps):
        val = jump_vals[idx]
        for i in range(segment_lengths[idx]):
            filtered_z[pos] = val
            pos += 1

    # Cleanup
    free(jump_idxs); free(jump_vals); free(segment_lengths)
    for i in range(order):
        free(gs[i])
    free(gs)

    return filtered_z