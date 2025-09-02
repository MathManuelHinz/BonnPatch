# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=True
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, log10, abs as c_abs, fabs
from libc.stdlib cimport malloc, free

# Type definitions for better performance
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t INT_t
ctypedef cnp.uint8_t BOOL_t

cdef inline double c_normalized_volume_deviation(double[::1] S, double[::1] M) nogil:
    """Optimized C version of normalized_volume_deviation"""
    cdef:
        Py_ssize_t i
        Py_ssize_t n = S.shape[0]
        double sum_diff = 0.0
        double sum_S = 0.0
        double sum_M = 0.0
        double diff
    
    for i in range(n):
        diff = S[i] * S[i] - M[i] * M[i]
        sum_diff += sqrt(c_abs(diff))
        sum_S += S[i]
        sum_M += M[i]
    
    return sum_diff / (sum_S + sum_M)

cdef inline double c_normalized_volume_deviation_2d(double[:, ::1] S, double[:, ::1] M) nogil:
    """Optimized C version for 2D arrays"""
    cdef:
        Py_ssize_t i, j
        Py_ssize_t rows = S.shape[0]
        Py_ssize_t cols = S.shape[1]
        double sum_diff = 0.0
        double sum_S = 0.0
        double sum_M = 0.0
        double diff
    
    for i in range(rows):
        for j in range(cols):
            diff = S[i, j] * S[i, j] - M[i, j] * M[i, j]
            sum_diff += sqrt(c_abs(diff))
            sum_S += S[i, j]
            sum_M += M[i, j]
    
    return sum_diff / (sum_S + sum_M)

def normalized_volume_deviation(S, M):
    """Python wrapper for normalized_volume_deviation"""
    cdef double[::1] S_view = S.astype(np.float64).ravel()
    cdef double[::1] M_view = M.astype(np.float64).ravel()
    return c_normalized_volume_deviation(S_view, M_view)

cdef inline Py_ssize_t c_digitize(double value, double[::1] bin_edges, Py_ssize_t res) nogil:
    """Fast C implementation of digitize for log-spaced bins"""
    cdef:
        Py_ssize_t low = 0
        Py_ssize_t high = res
        Py_ssize_t mid
        double log_value = log10(value)
    
    # Binary search for log-spaced bins
    while low < high:
        mid = (low + high) // 2
        if log10(bin_edges[mid]) <= log_value:
            low = mid + 1
        else:
            high = mid
    
    # Clamp to valid range
    if low > res - 1:
        return -1
    if low < 0:
        return -1
    return low - 1

cdef inline double c_hellinger_distance_1d(double[::1] mjp_1, double[::1] mjp_0) nogil:
    """Fast C implementation of Hellinger distance for 1D arrays"""
    cdef:
        Py_ssize_t i
        Py_ssize_t n = mjp_1.shape[0]
        double sum_1 = 0.0
        double sum_0 = 0.0
        double distance = 0.0
        double sqrt_p1, sqrt_p0, diff
        double norm_factor = 1.0 / sqrt(2.0)
    
    # Compute sums for normalization
    for i in range(n):
        sum_1 += mjp_1[i]
        sum_0 += mjp_0[i]
    
    # Avoid division by zero
    if sum_1 == 0.0 or sum_0 == 0.0:
        return sqrt(2.0)
    
    # Compute Hellinger distance
    for i in range(n):
        sqrt_p1 = sqrt(mjp_1[i] / sum_1)
        sqrt_p0 = sqrt(mjp_0[i] / sum_0)
        diff = sqrt_p1 - sqrt_p0
        distance += diff * diff
    
    return sqrt(distance) * norm_factor

cdef inline double c_hellinger_distance_2d(double[:, ::1] mjp_1, double[:, ::1] mjp_0) nogil:
    """
    Fast C implementation of Hellinger distance for 2D matrices
    Treats the entire matrix as a flattened probability distribution
    """
    cdef:
        Py_ssize_t i, j
        Py_ssize_t rows = mjp_1.shape[0]
        Py_ssize_t cols = mjp_1.shape[1]
        double sum_1 = 0.0
        double sum_0 = 0.0
        double distance = 0.0
        double sqrt_p1, sqrt_p0, diff
        double norm_factor = 1.0 / sqrt(2.0)
    
    # Compute total sums for normalization
    for i in range(rows):
        for j in range(cols):
            sum_1 += mjp_1[i, j]
            sum_0 += mjp_0[i, j]
    
    # Avoid division by zero
    if sum_1 == 0.0 or sum_0 == 0.0:
        return sqrt(2.0)
    
    # Compute Hellinger distance across all matrix elements
    for i in range(rows):
        for j in range(cols):
            sqrt_p1 = sqrt(mjp_1[i, j] / sum_1)
            sqrt_p0 = sqrt(mjp_0[i, j] / sum_0)
            diff = sqrt_p1 - sqrt_p0
            distance += diff * diff
    
    return sqrt(distance) * norm_factor

cdef inline double c_l1_distance_1d(double[::1] mjp_1, double[::1] mjp_0) nogil:
    """Fast C implementation of L1 (Manhattan) distance for 1D arrays"""
    cdef:
        Py_ssize_t i
        Py_ssize_t n = mjp_1.shape[0]
        double sum_1 = 0.0
        double sum_0 = 0.0
        double distance = 0.0
        double p1_norm, p0_norm
    
    # Compute sums for normalization
    for i in range(n):
        sum_1 += mjp_1[i]
        sum_0 += mjp_0[i]
    
    # Avoid division by zero
    if sum_1 == 0.0 and sum_0 == 0.0:
        return 0.0
    if sum_1 == 0.0 or sum_0 == 0.0:
        return 2.0  # Maximum L1 distance for normalized distributions
    
    # Compute L1 distance
    for i in range(n):
        p1_norm = mjp_1[i] / sum_1
        p0_norm = mjp_0[i] / sum_0
        distance += fabs(p1_norm - p0_norm)
    
    return distance

cdef inline double c_l1_distance_2d(double[:, ::1] mjp_1, double[:, ::1] mjp_0) nogil:
    """
    Fast C implementation of L1 distance for 2D matrices
    Treats the entire matrix as a flattened probability distribution
    """
    cdef:
        Py_ssize_t i, j
        Py_ssize_t rows = mjp_1.shape[0]
        Py_ssize_t cols = mjp_1.shape[1]
        double sum_1 = 0.0
        double sum_0 = 0.0
        double distance = 0.0
        double p1_norm, p0_norm
    
    # Compute total sums for normalization
    for i in range(rows):
        for j in range(cols):
            sum_1 += mjp_1[i, j]
            sum_0 += mjp_0[i, j]
    
    # Avoid division by zero
    if sum_1 == 0.0 and sum_0 == 0.0:
        return 0.0
    if sum_1 == 0.0 or sum_0 == 0.0:
        return 2.0  # Maximum L1 distance for normalized distributions
    
    # Compute L1 distance across all matrix elements
    for i in range(rows):
        for j in range(cols):
            p1_norm = mjp_1[i, j] / sum_1
            p0_norm = mjp_0[i, j] / sum_0
            distance += fabs(p1_norm - p0_norm)
    
    return distance

def get_histogram(double[::1] ts, BOOL_t[::1] ys, Py_ssize_t res=60, Py_ssize_t min_exp=-5,Py_ssize_t max_exp=1):
    """Optimized Cython version of get_histogram"""
    cdef:
        Py_ssize_t i, idx_o, idx_c
        Py_ssize_t n = ts.shape[0]
        BOOL_t current_state = ys[0]
        double last_to = 0.0
        double last_tc = 0.0
        double dt
        
    # Pre-allocate arrays
    cdef double[::1] h_o = np.zeros(res, dtype=np.float64)
    cdef double[::1] h_c = np.zeros(res, dtype=np.float64)
    cdef double[:, ::1] h_oc = np.zeros((res, res), dtype=np.float64)
    cdef double[:, ::1] h_co = np.zeros((res, res), dtype=np.float64)
    
    # Create bin edges using vectorized approach
    bin_edges_np = np.logspace(min_exp, max_exp, num=res+1)
    cdef double[::1] bin_edges = bin_edges_np
    
    # Main loop
    for i in range(1, n - 1):
        dt = ts[i] - ts[i-1]
        
        if ys[i] == current_state:
            if ys[i]:
                last_to += dt
            else:
                last_tc += dt
        else:
            current_state = ys[i]
            if ys[i]:  # now 1, previously 0
                last_tc += dt
                idx_c = c_digitize(last_tc, bin_edges, res)
                if idx_c>=0: h_c[idx_c] += 1.0
                if last_to > 0:
                    idx_o = c_digitize(last_to, bin_edges, res)
                    if idx_o>=0 and idx_c>=0: h_oc[idx_o, idx_c] += 1.0
                last_to = 0.0
            else:  # now 0, previously 1
                last_to += dt
                idx_o = c_digitize(last_to, bin_edges, res)
                if idx_o>=0: h_o[idx_o] += 1.0
                if last_tc > 0:
                    idx_c = c_digitize(last_tc, bin_edges, res)
                    if idx_c>=0 and idx_o>=0:h_co[idx_c, idx_o] += 1.0
                last_tc = 0.0
            
    
    return (np.asarray(h_o), np.asarray(h_c), 
            np.asarray(h_oc), np.asarray(h_co), 
            bin_edges_np)

def get_histogram_and_deviation(double[::1] ts, BOOL_t[::1] ys, Z, Py_ssize_t res=60, Py_ssize_t min_exp=-5,Py_ssize_t max_exp=1):
    """Optimized Cython version of get_histogram_and_deviation"""
    cdef:
        Py_ssize_t i, j, idx_o, idx_c
        Py_ssize_t n = ts.shape[0]
        BOOL_t current_state = ys[0]
        double last_to = 0.0
        double last_tc = 0.0
        double dt
        
    # Pre-allocate arrays
    cdef double[::1] h_o = np.zeros(res, dtype=np.float64)
    cdef double[::1] h_c = np.zeros(res, dtype=np.float64)
    cdef double[:, ::1] h_oc = np.zeros((res, res), dtype=np.float64)
    cdef double[:, ::1] h_co = np.zeros((res, res), dtype=np.float64)
    
    # Create bin edges and centers
    bin_edges_np = np.logspace(min_exp, max_exp, num=res+1)
    cdef double[::1] bin_edges = bin_edges_np
    bin_centers_np = 0.5 * (bin_edges_np[:-1] + bin_edges_np[1:])
    density_np = np.diff(bin_edges_np)
    
    # Pre-compute theoretical histograms
    cdef double[::1] theoretical_histogram_o = np.array([Z.f_o(t)*density_np[k] for k, t in enumerate(bin_centers_np)], dtype=np.float64)
    cdef double[::1] theoretical_histogram_c = np.array([Z.f_c(t)*density_np[k] for k, t in enumerate(bin_centers_np)], dtype=np.float64)
    cdef double[:, ::1] theoretical_histogram_oc = np.array([[Z.f_oc(t,s)*density_np[k]*density_np[l] for k, t in enumerate(bin_centers_np)] for l, s in enumerate(bin_centers_np)], dtype=np.float64)
    cdef double[:, ::1] theoretical_histogram_co = np.array([[Z.f_co(s,t)*density_np[l]*density_np[k] for l, s in enumerate(bin_centers_np)] for k, t in enumerate(bin_centers_np)], dtype=np.float64)
    
    # Initialize deviation storage
    deviations = {"o": [], "c": [], "oc": [], "co": []}
    
    # Pre-allocate temporary arrays for deviation computation
    cdef double[::1] scaled_theo_o = np.zeros(res, dtype=np.float64)
    cdef double[::1] scaled_theo_c = np.zeros(res, dtype=np.float64)
    cdef double[:, ::1] scaled_theo_oc = np.zeros((res, res), dtype=np.float64)
    cdef double[:, ::1] scaled_theo_co = np.zeros((res, res), dtype=np.float64)
    cdef double h_o_sum = 0.0, h_c_sum = 0.0, h_oc_sum = 0.0, h_co_sum = 0.0
    
    # Main loop with progress tracking
    for i in range(1, n):
        dt = ts[i] - ts[i-1]
        
        if ys[i] == current_state:
            if ys[i]:
                last_to += dt
            else:
                last_tc += dt
        else:
            if ys[i]:  # now 1, previously 0
                last_tc += dt
                idx_c = c_digitize(last_tc, bin_edges, res)
                h_c[idx_c] += 1.0
                if last_to > 0:
                    idx_o = c_digitize(last_to, bin_edges, res)
                    h_oc[idx_c, idx_o] += 1.0
                last_to = 0.0
            else:  # now 0, previously 1
                last_to += dt
                idx_o = c_digitize(last_to, bin_edges, res)
                h_o[idx_o] += 1.0
                if last_tc > 0:
                    idx_c = c_digitize(last_tc, bin_edges, res)
                    h_co[idx_o, idx_c] += 1.0
                last_tc = 0.0
            current_state = ys[i]
        
        # Compute deviations periodically (every 100 iterations after initial phase)
        if i > 4:
            # Compute sums for scaling
            h_o_sum = 0.0
            h_c_sum = 0.0
            h_oc_sum = 0.0
            h_co_sum = 0.0
            
            for j in range(res):
                h_o_sum += h_o[j]
                h_c_sum += h_c[j]
                scaled_theo_o[j] = h_o_sum * theoretical_histogram_o[j]
                scaled_theo_c[j] = h_c_sum * theoretical_histogram_c[j]
                for k in range(res):
                    h_oc_sum += h_oc[j, k]
                    h_co_sum += h_co[j, k]
            
            for j in range(res):
                for k in range(res):
                    scaled_theo_oc[j, k] = h_oc_sum * theoretical_histogram_oc[j, k]
                    scaled_theo_co[j, k] = h_co_sum * theoretical_histogram_co[j, k]
            
            # Compute deviations using optimized C functions
            deviations["o"].append(c_normalized_volume_deviation(h_o, scaled_theo_o))
            deviations["c"].append(c_normalized_volume_deviation(h_c, scaled_theo_c))
            deviations["oc"].append(c_normalized_volume_deviation_2d(h_oc, scaled_theo_oc))
            deviations["co"].append(c_normalized_volume_deviation_2d(h_co, scaled_theo_co))
    
    return (np.asarray(h_o), np.asarray(h_c), 
            np.asarray(h_oc), np.asarray(h_co), 
            bin_edges_np, deviations)

def get_histogram_and_hellinger(double[::1] ts, BOOL_t[::1] ys, Z, Py_ssize_t res=60, Py_ssize_t min_exp=-5,Py_ssize_t max_exp=1):
    """Optimized Cython version of get_histogram_and_deviation"""
    cdef:
        Py_ssize_t i, j, idx_o, idx_c
        Py_ssize_t n = ts.shape[0]
        BOOL_t current_state = ys[0]
        double last_to = 0.0
        double last_tc = 0.0
        double dt
        
    # Pre-allocate arrays
    cdef double[::1] h_o = np.zeros(res, dtype=np.float64)
    cdef double[::1] h_c = np.zeros(res, dtype=np.float64)
    cdef double[:, ::1] h_oc = np.zeros((res, res), dtype=np.float64)
    cdef double[:, ::1] h_co = np.zeros((res, res), dtype=np.float64)
    
    # Create bin edges and centers
    bin_edges_np = np.logspace(min_exp, max_exp, num=res+1)
    cdef double[::1] bin_edges = bin_edges_np
    bin_centers_np = 0.5 * (bin_edges_np[:-1] + bin_edges_np[1:])
    density_np = np.diff(bin_edges_np)
    
    # Pre-compute theoretical histograms
    cdef double[::1] theoretical_histogram_o = np.array([Z.f_o(t)*density_np[k] for k, t in enumerate(bin_centers_np)], dtype=np.float64)
    cdef double[::1] theoretical_histogram_c = np.array([Z.f_c(t)*density_np[k] for k, t in enumerate(bin_centers_np)], dtype=np.float64)
    cdef double[:, ::1] theoretical_histogram_oc = np.array([[Z.f_oc(t,s)*density_np[k]*density_np[l] for k, t in enumerate(bin_centers_np)] for l, s in enumerate(bin_centers_np)], dtype=np.float64)
    cdef double[:, ::1] theoretical_histogram_co = np.array([[Z.f_co(s,t)*density_np[l]*density_np[k] for l, s in enumerate(bin_centers_np)] for k, t in enumerate(bin_centers_np)], dtype=np.float64)
    
    # Initialize deviation storage
    deviations = {"o": [], "c": [], "oc": [], "co": []}
    
    # Pre-allocate temporary arrays for deviation computation
    cdef double[::1] scaled_theo_o = np.zeros(res, dtype=np.float64)
    cdef double[::1] scaled_theo_c = np.zeros(res, dtype=np.float64)
    cdef double[:, ::1] scaled_theo_oc = np.zeros((res, res), dtype=np.float64)
    cdef double[:, ::1] scaled_theo_co = np.zeros((res, res), dtype=np.float64)
    cdef double h_o_sum = 0.0, h_c_sum = 0.0, h_oc_sum = 0.0, h_co_sum = 0.0
    
    # Main loop with progress tracking
    for i in range(1, n):
        dt = ts[i] - ts[i-1]
        
        if ys[i] == current_state:
            if ys[i]:
                last_to += dt
            else:
                last_tc += dt
        else:
            if ys[i]:  # now 1, previously 0
                last_tc += dt
                idx_c = c_digitize(last_tc, bin_edges, res)
                h_c[idx_c] += 1.0
                if last_to > 0:
                    idx_o = c_digitize(last_to, bin_edges, res)
                    h_oc[idx_c, idx_o] += 1.0
                last_to = 0.0
            else:  # now 0, previously 1
                last_to += dt
                idx_o = c_digitize(last_to, bin_edges, res)
                h_o[idx_o] += 1.0
                if last_tc > 0:
                    idx_c = c_digitize(last_tc, bin_edges, res)
                    h_co[idx_o, idx_c] += 1.0
                last_tc = 0.0
            current_state = ys[i]
        
        # Compute deviations periodically (every 100 iterations after initial phase)
        if i > 4:
            # Compute sums for scaling
            h_o_sum = 0.0
            h_c_sum = 0.0
            h_oc_sum = 0.0
            h_co_sum = 0.0
            
            for j in range(res):
                h_o_sum += h_o[j]
                h_c_sum += h_c[j]
                scaled_theo_o[j] = h_o_sum * theoretical_histogram_o[j]
                scaled_theo_c[j] = h_c_sum * theoretical_histogram_c[j]
                for k in range(res):
                    h_oc_sum += h_oc[j, k]
                    h_co_sum += h_co[j, k]
            
            for j in range(res):
                for k in range(res):
                    scaled_theo_oc[j, k] = h_oc_sum * theoretical_histogram_oc[j, k]
                    scaled_theo_co[j, k] = h_co_sum * theoretical_histogram_co[j, k]
            
            # Compute deviations using optimized C functions
            deviations["o"].append(c_hellinger_distance_1d(h_o, scaled_theo_o))
            deviations["c"].append(c_hellinger_distance_1d(h_c, scaled_theo_c))
            deviations["oc"].append(c_hellinger_distance_2d(h_oc, scaled_theo_oc))
            deviations["co"].append(c_hellinger_distance_2d(h_co, scaled_theo_co))
    
    return (np.asarray(h_o), np.asarray(h_c), 
            np.asarray(h_oc), np.asarray(h_co), 
            bin_edges_np, deviations)

def get_histogram_and_l1(double[::1] ts, BOOL_t[::1] ys, Z, Py_ssize_t res=60, Py_ssize_t min_exp=-5,Py_ssize_t max_exp=1):
    """Optimized Cython version of get_histogram_and_deviation"""
    cdef:
        Py_ssize_t i, j, idx_o, idx_c
        Py_ssize_t n = ts.shape[0]
        BOOL_t current_state = ys[0]
        double last_to = 0.0
        double last_tc = 0.0
        double dt
        
    # Pre-allocate arrays
    cdef double[::1] h_o = np.zeros(res, dtype=np.float64)
    cdef double[::1] h_c = np.zeros(res, dtype=np.float64)
    cdef double[:, ::1] h_oc = np.zeros((res, res), dtype=np.float64)
    cdef double[:, ::1] h_co = np.zeros((res, res), dtype=np.float64)
    
    # Create bin edges and centers
    bin_edges_np = np.logspace(min_exp, max_exp, num=res+1)
    cdef double[::1] bin_edges = bin_edges_np
    bin_centers_np = 0.5 * (bin_edges_np[:-1] + bin_edges_np[1:])
    density_np = np.diff(bin_edges_np)
    
    # Pre-compute theoretical histograms
    cdef double[::1] theoretical_histogram_o = np.array([Z.f_o(t)*density_np[k] for k, t in enumerate(bin_centers_np)], dtype=np.float64)
    cdef double[::1] theoretical_histogram_c = np.array([Z.f_c(t)*density_np[k] for k, t in enumerate(bin_centers_np)], dtype=np.float64)
    cdef double[:, ::1] theoretical_histogram_oc = np.array([[Z.f_oc(t,s)*density_np[k]*density_np[l] for k, t in enumerate(bin_centers_np)] for l, s in enumerate(bin_centers_np)], dtype=np.float64)
    cdef double[:, ::1] theoretical_histogram_co = np.array([[Z.f_co(s,t)*density_np[l]*density_np[k] for l, s in enumerate(bin_centers_np)] for k, t in enumerate(bin_centers_np)], dtype=np.float64)
    
    # Initialize deviation storage
    deviations = {"o": [], "c": [], "oc": [], "co": []}
    
    # Pre-allocate temporary arrays for deviation computation
    cdef double[::1] scaled_theo_o = np.zeros(res, dtype=np.float64)
    cdef double[::1] scaled_theo_c = np.zeros(res, dtype=np.float64)
    cdef double[:, ::1] scaled_theo_oc = np.zeros((res, res), dtype=np.float64)
    cdef double[:, ::1] scaled_theo_co = np.zeros((res, res), dtype=np.float64)
    cdef double h_o_sum = 0.0, h_c_sum = 0.0, h_oc_sum = 0.0, h_co_sum = 0.0
    
    # Main loop with progress tracking
    for i in range(1, n):
        dt = ts[i] - ts[i-1]
        
        if ys[i] == current_state:
            if ys[i]:
                last_to += dt
            else:
                last_tc += dt
        else:
            if ys[i]:  # now 1, previously 0
                last_tc += dt
                idx_c = c_digitize(last_tc, bin_edges, res)
                h_c[idx_c] += 1.0
                if last_to > 0:
                    idx_o = c_digitize(last_to, bin_edges, res)
                    h_oc[idx_c, idx_o] += 1.0
                last_to = 0.0
            else:  # now 0, previously 1
                last_to += dt
                idx_o = c_digitize(last_to, bin_edges, res)
                h_o[idx_o] += 1.0
                if last_tc > 0:
                    idx_c = c_digitize(last_tc, bin_edges, res)
                    h_co[idx_o, idx_c] += 1.0
                last_tc = 0.0
            current_state = ys[i]
        
        # Compute deviations periodically (every 100 iterations after initial phase)
        if i > 4:
            # Compute sums for scaling
            h_o_sum = 0.0
            h_c_sum = 0.0
            h_oc_sum = 0.0
            h_co_sum = 0.0
            
            for j in range(res):
                h_o_sum += h_o[j]
                h_c_sum += h_c[j]
                scaled_theo_o[j] = h_o_sum * theoretical_histogram_o[j]
                scaled_theo_c[j] = h_c_sum * theoretical_histogram_c[j]
                for k in range(res):
                    h_oc_sum += h_oc[j, k]
                    h_co_sum += h_co[j, k]
            
            for j in range(res):
                for k in range(res):
                    scaled_theo_oc[j, k] = h_oc_sum * theoretical_histogram_oc[j, k]
                    scaled_theo_co[j, k] = h_co_sum * theoretical_histogram_co[j, k]
            
            # Compute deviations using optimized C functions
            deviations["o"].append(c_l1_distance_1d(h_o, scaled_theo_o))
            deviations["c"].append(c_l1_distance_1d(h_c, scaled_theo_c))
            deviations["oc"].append(c_l1_distance_2d(h_oc, scaled_theo_oc))
            deviations["co"].append(c_l1_distance_2d(h_co, scaled_theo_co))
    
    return (np.asarray(h_o), np.asarray(h_c), 
            np.asarray(h_oc), np.asarray(h_co), 
            bin_edges_np, deviations)
