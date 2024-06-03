"""
Defines CUDA kernels written in Numba API. 
"""

import warnings
from typing import Literal, Optional
import numba
from numba import cuda, float32, float64, int32, types, void


def cosine_greedy_kernel(
    tolerance: float = 0.1,
    shift: float = 0,
    mz_power: float = 0.0,
    int_power: float = 1.0,
    match_limit: int = 1024,
    batch_size: int = 2048,
    n_max_peaks: int = 2048,
    is_symmetric: bool = False,
    precursor_shift: bool = False,
    spectra_dtype: Optional[str] = None,
) -> callable:
    """
    Compiles and returns a CUDA kernel function for calculating cosine similarity scores between spectra.

    Parameters:
    -----------
    tolerance : float, optional
        Tolerance parameter for m/z matching, by default 0.1.
    shift : float, optional
        Shift parameter for m/z matching, by default 0.
    mz_power : float, optional
        Power parameter for m/z intensity calculation, by default 0.
    int_power : float, optional
        Power parameter for intensity calculation, by default 1.
    match_limit : int, optional
        Maximum number of matches to consider, by default 1024.
    batch_size : int, optional
        Batch size for processing spectra, by default 2048.
    n_max_peaks : int, optional
        Maximum number of peaks to consider, by default 2048.
    is_symmetric : bool, optional
        Flag indicating if the similarity matrix is symmetric, by default False.
    dtype:
        By default, we use float32 for greedy cosine and float64 for modified cosine. This can be overridden. Accepts 
        arguments of type `numba.types.Float`
    Returns:
    --------
    callable
        CUDA kernel function for calculating cosine similarity scores.
    """

    if is_symmetric:
        warnings.warn("no effect from is_symmetric, it is not yet implemented")

    # Define global constants. These values will be
    # transferred by NUMBA to the GPU, global read-only constants
    PRECURSOR_SHIFT = precursor_shift
    if PRECURSOR_SHIFT:
        MATCH_LIMIT = match_limit * 2
    else:
        MATCH_LIMIT = match_limit
    SHIFT = shift
    MZ_POWER = mz_power
    INT_POWER = int_power
    TOLERANCE = tolerance
    N_MAX_PEAKS = n_max_peaks
    R, Q = batch_size, batch_size
    THREADS_PER_BLOCK_X = 1
    THREADS_PER_BLOCK_Y = 512
    THREADS_PER_BLOCK = (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)
    BLOCKS_PER_GRID_X = (R + THREADS_PER_BLOCK_X - 1) // THREADS_PER_BLOCK_X
    BLOCKS_PER_GRID_Y = (Q + THREADS_PER_BLOCK_Y - 1) // THREADS_PER_BLOCK_Y
    BLOCKS_PER_GRID = (BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y)

    # It is recommended to use float64 for modified cosine calculations
    # since the error can be 
    if PRECURSOR_SHIFT:
        FLOAT = numba.float64
    else:
        FLOAT = numba.float32

    # If the user forces the dtype, we oblige
    if spectra_dtype is not None:
        FLOAT = numba.types.Float(spectra_dtype)

    # Arguments: matched peaks [match_limit], peak value [match_limit], and return values [2]
    @cuda.jit(int32(int32, int32, FLOAT[:,:,::1], FLOAT[:,:,::1], FLOAT[:,::1], int32[::1], FLOAT[::1]), device=True, inline=True)
    def collect_peak_pairs(
            # Inputs
            reference_i, query_j, rspec, qspec, metadata,
            # Outputs
            matches, values,
        ):
        """
        Roughly equivalent to the `matchms.similarity.spectrum_similarity_functions.find_matches`
        Collects all matching peaks within TOLERANCE and writes found `matches` and their peak products in `values`.
        Returns one integer `num_matches`, which specifies how many peaks we've managed to match.
        """
        rleni = types.int32(metadata[0, reference_i])
        qlenj = types.int32(metadata[1, query_j])

        # If either R or Q spectra are empty, return
        if rleni == 0 or qlenj == 0:
            return 0

        # Unpack R into m/z and intensity sets
        rmz = rspec[0]  # rmz is [batch_size, n_max_peaks]
        rint = rspec[1]  # rint is [batch_size, n_max_peaks]

        # Read current thread's reference spectrum
        spec1_mz = rmz[reference_i]  # spec1_mz is [n_max_peaks]
        spec1_int = rint[reference_i]  # spec1_int is [n_max_peaks]

        # Similar steps for reading this thread's Query
        qmz = qspec[0]
        qint = qspec[1]
        spec2_mz = qmz[query_j]
        spec2_int = qint[query_j]

        lowest_idx = types.int32(0)
        num_match = types.int32(0)
        overflow = types.boolean(False)

        for peak1_idx in range(rleni):
            if overflow:
                break
            mz_r = spec1_mz[peak1_idx]
            int_r = spec1_int[peak1_idx]
            for peak2_idx in range(lowest_idx, qlenj):
                if overflow:
                    break
                mz_q = spec2_mz[peak2_idx]
                if mz_q + SHIFT > mz_r + TOLERANCE:
                    break
                if mz_q + SHIFT + TOLERANCE < mz_r:
                    lowest_idx = peak2_idx + 1
                else:
                    if not overflow:
                        int_q = spec2_int[peak2_idx]
                        # Binary trick!
                        # since we know that the largest imaginable peak index can fit in 13 bits
                        # We pack two 16bit ints in 32bit int to use less memory
                        matches[num_match] = (peak1_idx << 16) | peak2_idx
                        values[num_match] = (mz_r**MZ_POWER * int_r**INT_POWER) * (
                            mz_q**MZ_POWER * int_q**INT_POWER
                        )
                        num_match += 1
                        # Once we have filled the entire allocated `matches` array, we stop adding more. We call this the `overflow`
                        overflow = (
                            num_match >= MATCH_LIMIT
                        )  # This is the errorcode for overflow
        if overflow:
            num_match = -num_match
        return num_match

    # Arguments: matched peaks [match_limit], peak value [match_limit], and return values [2]
    @cuda.jit(int32(int32, int32, FLOAT[:,:,::1], FLOAT[:,:,::1], FLOAT[:,::1], int32[::1], FLOAT[::1]), device=True, inline=True)
    def collect_shifted_peak_pairs(
            # Inputs
            reference_i, query_j, rspec, qspec, metadata,
            # Outputs
            matches, values,
        ):
        rleni = types.int32(metadata[0, reference_i])
        qlenj = types.int32(metadata[1, query_j])

        # If either R or Q spectra are empty, return
        if rleni == 0 or qlenj == 0:
            return 0

        # Unpack R into m/z and intensity sets
        rmz = rspec[0]  # rmz is [batch_size, n_max_peaks]
        rint = rspec[1]  # rint is [batch_size, n_max_peaks]

        # Read current thread's reference spectrum
        spec1_mz = rmz[reference_i]  # spec1_mz is [n_max_peaks]
        spec1_int = rint[reference_i]  # spec1_int is [n_max_peaks]

        # Similar steps for reading this thread's Query
        qmz = qspec[0]
        qint = qspec[1]
        spec2_mz = qmz[query_j]
        spec2_int = qint[query_j]

        lowest_idx = types.int32(0)
        num_match = types.int32(0)
        overflow = types.boolean(False)

        for peak1_idx in range(rleni):
            if overflow:
                break
            mz_r = spec1_mz[peak1_idx]
            int_r = spec1_int[peak1_idx]
            for peak2_idx in range(lowest_idx, qlenj):
                if overflow:
                    break
                mz_q = spec2_mz[peak2_idx]
                if mz_q + SHIFT > mz_r + TOLERANCE:
                    break
                if mz_q + SHIFT + TOLERANCE < mz_r:
                    lowest_idx = peak2_idx + 1
                else:
                    if not overflow:
                        int_q = spec2_int[peak2_idx]
                        # Binary trick!
                        # since we know that the largest imaginable peak index can fit in 13 bits
                        # We pack two 16bit ints in 32bit int to use less memory
                        matches[num_match] = (peak1_idx << 16) | peak2_idx
                        values[num_match] = (mz_r**MZ_POWER * int_r**INT_POWER) * (
                            mz_q**MZ_POWER * int_q**INT_POWER
                        )
                        num_match += 1
                        # Once we have filled the entire allocated `matches` array, we stop adding more. We call this the `overflow`
                        overflow = (
                            num_match >= (MATCH_LIMIT//2)
                        )  # This is the errorcode for overflow

        lowest_idx = types.int32(0)
        overflow_shifted = types.boolean(False)
        # shift = metadata[4, reference_i] - metadata[5, query_j]
        rpmz = metadata[4, reference_i]
        qpmz = metadata[5, query_j]
        # r_pmz = 
        for peak1_idx in range(rleni):
            if overflow_shifted:
                break
            mz_r = spec1_mz[peak1_idx]
            int_r = spec1_int[peak1_idx]
            for peak2_idx in range(lowest_idx, qlenj):
                if overflow_shifted:
                    break
                mz_q = spec2_mz[peak2_idx]
                if mz_q + rpmz > mz_r + qpmz + TOLERANCE:
                    break
                if mz_q + rpmz + TOLERANCE < mz_r + qpmz:
                    lowest_idx = peak2_idx + 1
                else:
                    if not overflow_shifted:
                        int_q = spec2_int[peak2_idx]
                        # Binary trick!
                        # since we know that the largest imaginable peak index can fit in 13 bits
                        # We pack two 16bit ints in 32bit int to use less memory
                        matches[num_match] = (peak1_idx << 16) | peak2_idx
                        values[num_match] = (mz_r**MZ_POWER * int_r**INT_POWER) * (
                            mz_q**MZ_POWER * int_q**INT_POWER
                        )
                        num_match += 1
                        # Once we have filled the entire allocated `matches` array, we stop adding more. We call this the `overflow`
                        overflow_shifted = (
                            num_match >= MATCH_LIMIT
                        )  # This is the errorcode for overflow
        if overflow or overflow_shifted:
            num_match = -num_match
        return num_match
    

    @cuda.jit(device=True, inline=True)
    def sort_peaks_by_value(matches, values, num_match):
        temp_matches = cuda.local.array(MATCH_LIMIT, types.int32)
        temp_values = cuda.local.array(MATCH_LIMIT, FLOAT)

        # k is an expanding window of sorting. We initially sort single (size-1) elements,
        # At this point, all 2-tuples in the array are themselves sorted. We then double the sorting window (to 2)
        # Since all 2-tuples are themselves sorted, we efficiently create size-4 sorted tuples from each.
        # This doubling continues, until all elements are in order. This takes O(nlog2(n)) time, where n is match_limit.
        k = types.int32(1)
        while k < num_match:
            for left in range(0, num_match - k, k * 2):
                rght = left + k
                rend = rght + k

                rend = min(rend, num_match)

                m = left
                ix = left
                jx = rght
                while ix < rght and jx < rend:
                    mask = values[ix] > values[jx]
                    temp_matches[m] = mask * matches[ix] + (1 - mask) * matches[jx]
                    temp_values[m] = mask * values[ix] + (1 - mask) * values[jx]
                    ix += mask
                    jx += 1 - mask
                    m += 1

                while ix < rght:
                    temp_matches[m] = matches[ix]
                    temp_values[m] = values[ix]
                    ix += 1
                    m += 1

                while jx < rend:
                    temp_matches[m] = matches[jx]
                    temp_values[m] = values[jx]
                    jx += 1
                    m += 1

                for m in range(left, rend):
                    matches[m] = temp_matches[m]
                    values[m] = temp_values[m]
            k *= 2
    # Arguments are: References [2 (m/z, intensity), batch_size, n_max_peaks], Queries [Same], 
    # lengths [2 (r length, q length), batch_size], norms [2 (r norm, q_norm), batch_size], outputs [3 (score, matches, overflow), batch_size, batch_size]
    # The signature float32[:,:,::1] means a 3D contiguous Float32 array, with row-major ordering.
    @cuda.jit(
        void(FLOAT[:,:,::1], FLOAT[:,:,::1], FLOAT[:,::1], float32[:,:,::1])
    )
    def _kernel(
        rspec,
        qspec,
        metadata,
        out,
    ):
        """
        CUDA kernel function that will be translated to GPU-executable machine code on the fly.

        Parameters:
        -----------
        rspec : cuda.devicearray
            All reference spectra. Float tensor.
            Shape [2, batch_size, n_max_peaks]. Zero-padded, when spectra are smaller than n_max_peaks.
            Stores mz (0th slice), and intensity (1st slice).
        qspec : cuda.devicearray
            Same structure as rspec.
        lens : cuda.devicearray
            Number of peaks in both spectra batches. Integer tensor. Shape [2, batch_size]. Zero padded, when number of spectra are not divisible by
            batch size and we are processing the edge-batches.
        norms : cuda.devicearray
            Contains a pre-computed norm for both spectra batches. Float tensor. Shape [2, batch_size].
        out : cuda.devicearray
            Stores results. Float tensor. Shape [batch_size, batch_size, 3].
            The last dimention (3) contains: score, matches, overflow.

        The kernel is designed to efficiently compute cosine similarity scores
        between reference and query spectra by relying on CUDA parallelization.

        It performs the following steps:
        1. Find potential peak matches between the spectra based on m/z tolerance.
        2. Sort matched peaks based on cosine product value.
        3. Accumulate unnormalized cosine score while discarding duplicate peaks, beginning with the largest pair.
        4. Divide unnormalized score by the pre-computed spectra norm.

        """
        ## PREAMBLE:

        # Get global indices
        i, j = cuda.grid(2)

        # Check we aren't out of the max possible grid size
        if i >= R or j >= Q:
            return

        # Set zeros, since we know we are in the grid
        out[0, i, j] = 0
        out[1, i, j] = 0
        out[2, i, j] = 0

        #### PART 1: Collect peak pairs  ####
        # Allocate matches and values arrays in GPU global memory.
        # These will store peak indices (r_idx, q_idx) and peak contributions (float) respectively

        matches = cuda.local.array(MATCH_LIMIT, types.int32)
        values = cuda.local.array(MATCH_LIMIT, FLOAT)

        ### FIND MATCHES
        if PRECURSOR_SHIFT:
            num_match = collect_shifted_peak_pairs(
                        i, j, rspec, qspec, metadata,
                        matches, values
                        )
        else:
            num_match = collect_peak_pairs(
                            i, j, rspec, qspec, metadata, 
                            matches, values
                        )
            
        # In case we didn't get any matches, we return. We already have set 0 as the default output above.
        if num_match == 0:
            return
        # We store the overflow flag inside num_match, when it's negative, it's overflown.
        if num_match < 0:
            out[2, i, j] = 1.0
        # Correct the negative regardless
        num_match = abs(num_match)

        #### PART 2: Sort peaks from largest to smallest ####
        # We use a non-recursive mergesort in order to sort matches by the peak contributions (values)
        # We require a 2 additional arrays to store the sorting intermediate results.
        sort_peaks_by_value(matches, values, num_match)

        #### PART 3: Sum unnormalized peak contributions, and remove duplicates ####
        # Having peak matches sorted we can start summing all peak contributions to the unnormalized score, from the largest to the smallest.
        # To avoid duplicates, we create two boolean arrays that keep track of used matches.
        used_r = cuda.local.array(N_MAX_PEAKS, types.boolean)
        used_q = cuda.local.array(N_MAX_PEAKS, types.boolean)
        for m in range(N_MAX_PEAKS):
            used_r[m] = False
            used_q[m] = False

        used_matches = types.float32(0)
        score = types.float32(0.0)

        for m in range(num_match):
            # The binary trick is undone to get both peak indices back
            c = matches[m]
            peak1_idx = c >> 16
            peak2_idx = c & 0x0000_FFFF

            if (not used_r[peak1_idx]) and (not used_q[peak2_idx]):
                used_r[peak1_idx] = True
                used_q[peak2_idx] = True
                score += values[m]
                used_matches += 1

        # We finally normalize and return the score

        # Read pre-calculated norms for R and Q, multiply to get the score norm
        score_norm = metadata[2, i] * metadata[3, j]

        out[0, i, j] = score / score_norm
        out[1, i, j] = used_matches

        #### Sorting + Accumulation path 2-3: ALTENRNATIVE SORT+ACCUMULATE PATHWAY ####
        # This pathway could be faster when matches and average scores are very rare.
        # One would compile and time both kernels, and retain the fastest

        # score = types.float32(0.0)
        # used_matches = types.uint16(0)
        # for _ in range(0, num_match):
        #     max_prod = types.float32(-1.0)
        #     max_peak1_idx = 0
        #     max_peak2_idx = 0

        #     for sj in range(0, num_match):
        #         c = matches[sj]
        #         if c != -1:
        #             peak1_idx = c >> 16
        #             peak2_idx = c & 0x0000_FFFF

        #             prod = values[sj]

        #             # > was changed to >= and that took 2 weeks... also finding that 'mergesort' in original similarity algorithm
        #             # is what can prevent instability.
        #             if prod >= max_prod:
        #                 max_prod = prod
        #                 max_peak1_idx = peak1_idx
        #                 max_peak2_idx = peak2_idx

        #     if max_prod != -1:
        #         for sj in range(0, num_match):
        #             c = matches[sj]
        #             if c != -1:
        #                 peak1_idx = c >> 16
        #                 peak2_idx = c & 0x0000_FFFF
        #                 if (peak1_idx == max_peak1_idx or peak2_idx == max_peak2_idx):
        #                     matches[sj] = -1 # "Remove" it
        #         score += max_prod
        #         used_matches += 1
        #     else:
        #         break

    def kernel(
        rspec,
        qspec,
        metadata,
        out,
        stream: cuda.stream = None,
    ) -> None:
        """
        Launches the CUDA kernel for calculating cosine similarity scores.

        Parameters:
        -----------
        rspec : cuda.devicearray
            Array containing reference spectra data.
        qspec_cu : cuda.devicearray
            Array containing query spectra data.
        lens_cu : cuda.devicearray
            Array containing lengths of spectra.
        out_cu : cuda.devicearray
            Precalculated norms for each R and Q
        stream : cuda.stream, optional
            CUDA stream for asynchronous execution, by default None.
        """
        
        _kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK, stream](
            rspec,
            qspec,
            metadata,
            out,
        )

    return kernel