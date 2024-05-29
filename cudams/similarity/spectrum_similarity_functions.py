"""
Defines CUDA kernels written in Numba API. 
"""
import warnings
from numba import cuda, types

def cosine_greedy_kernel(
    tolerance: float = 0.1,
    shift: float = 0,
    mz_power: float = 0.0,
    int_power: float = 1.0,
    match_limit: int = 1024,
    batch_size: int = 2048,
    n_max_peaks: int = 2048,
    is_symmetric: bool = False,
):
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

    Returns:
    --------
    callable
        CUDA kernel function for calculating cosine similarity scores.
    """

    if is_symmetric:
        warnings.warn("no effect from is_symmetric, it is not yet implemented")

    MATCH_LIMIT = match_limit
    N_MAX_PEAKS = n_max_peaks
    R, Q = batch_size, batch_size
    THREADS_PER_BLOCK_X = 1
    THREADS_PER_BLOCK_Y = 512
    THREADS_PER_BLOCK = (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)
    BLOCKS_PER_GRID_X = (R + THREADS_PER_BLOCK_X - 1) // THREADS_PER_BLOCK_X
    BLOCKS_PER_GRID_Y = (Q + THREADS_PER_BLOCK_Y - 1) // THREADS_PER_BLOCK_Y
    BLOCKS_PER_GRID = (BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y)

    @cuda.jit('void(float32[:,:,:], float32[:,:,:], int32[:,:], float32[:,:], float32[:,:,:])')
    def _kernel(
        rspec,
        qspec,
        lens,
        norms,
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

        # Get actual number of peaks in R
        rleni = lens[0, i]
        qlenj = lens[1, j]
        
        # If either R or Q spectra are empty, return
        if rleni == 0 or qlenj == 0:
            return
        
        # Read pre-calculated norms for R and Q, multiply to get the score norm
        score_norm = norms[0, i] * norms[1, j]

        # Unpack R into m/z and intensity sets 
        rmz = rspec[0] # rmz is [batch_size, n_max_peaks]
        rint = rspec[1] # rint is [batch_size, n_max_peaks]

        # Read current thread's reference spectrum
        spec1_mz = rmz[i] # spec1_mz is [n_max_peaks]
        spec1_int = rint[i] # spec1_int is [n_max_peaks]

        # Similar steps for reading this thread's Query
        qmz = qspec[0]
        qint = qspec[1]
        spec2_mz = qmz[j]
        spec2_int = qint[j]

        #### PART 1: Collect peak pairs  ####
        # Allocate matches and values arrays in GPU global memory. 
        # These will store peak indices (r_idx, q_idx) and peak contributions (float) respectively

        matches = cuda.local.array(MATCH_LIMIT, types.int32)
        values = cuda.local.array(MATCH_LIMIT, types.float32)

        # We follow the matchms.similarity.spectrum_similarity_functions.collect_peak_pairs as closely as possible
        
        ### FIND MATCHES 
        # The following is equivalent to the `matchms.similarity.spectrum_similarity_functions.find_matches`
        # We do a compare all peaks in both spectra, and keep track of peaks within the given tolerance.
        # We store first `match_limit` number of matches and peak contributions (values) into `matches` and `values` arrays.
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
                if mz_q + shift > mz_r + tolerance:
                    break
                if mz_q + shift + tolerance < mz_r:
                    lowest_idx = peak2_idx + 1
                else:
                    if not overflow:
                        int_q = spec2_int[peak2_idx]
                        # Binary trick! 
                        # since we know that the largest imaginable peak index can fit in 13 bits 
                        # We pack two 16bit ints in 32bit int to use less memory
                        matches[num_match] = (peak1_idx << 16) | peak2_idx
                        values[num_match] = (mz_r ** mz_power * int_r ** int_power) * (mz_q ** mz_power * int_q ** int_power)
                        num_match += 1
                        # Once we have filled the entire allocated `matches` array, we stop adding more. We call this the `overflow`
                        overflow = num_match >= MATCH_LIMIT  # This is the errorcode for overflow

        # The overflow gets returned as the 3rd output for this RxQ comparison.
        if overflow:
            out[2, i, j] = 1.0
        
        # In case we didn't get any matches, we return. We already have set 0 as the default output above.
        if num_match == 0:
            return

        #### PART 2: Sort peaks from largest to smallest ####
        # We use a non-recursive mergesort in order to sort matches by the peak contributions (values)
        # We require a 2 additional arrays to store the sorting intermediate results.
        temp_matches = cuda.local.array(MATCH_LIMIT, types.int32)
        temp_values = cuda.local.array(MATCH_LIMIT, types.float32)

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

                m = left; ix = left; jx = rght;
                while ix < rght and jx < rend:
                    mask = (values[ix] > values[jx])
                    temp_matches[m] = mask * matches[ix] + (1 - mask) * matches[jx]
                    temp_values[m] = mask * values[ix] + (1 - mask) * values[jx]
                    ix+=mask
                    jx+=(1-mask)
                    m+=1

                while ix < rght:
                    temp_matches[m] = matches[ix]; 
                    temp_values[m] = values[ix]; 
                    ix+=1; m+=1;
                
                while jx < rend:
                    temp_matches[m] = matches[jx]; 
                    temp_values[m] = values[jx]; 
                    jx+=1; m+=1;
                
                for m in range(left, rend):
                    matches[m] = temp_matches[m]; 
                    values[m] = temp_values[m]; 
            k *= 2

        #### PART 3: Sum unnormalized peak contributions, and remove duplicates ####
        # Having peak matches sorted we can start summing all peak contributions to the unnormalized score, from the largest to the smallest.
        # To avoid duplicates, we create two boolean arrays that keep track of used matches.
        used_r = cuda.local.array(N_MAX_PEAKS, types.boolean)
        used_q = cuda.local.array(N_MAX_PEAKS, types.boolean)
        for m in range(N_MAX_PEAKS):
            used_r[m] = False
            used_q[m] = False

        used_matches = 0
        score = 0.0

        for m in range(num_match):
            # The binary trick is undone to get both peak indices back
            c = matches[m]
            peak1_idx = c >> 16
            peak2_idx = c & 0x0000_FFFF

            if (not used_r[peak1_idx]) and (not used_q[peak2_idx]):
                used_r[peak1_idx] = True
                used_q[peak2_idx] = True
                score += values[m];
                used_matches += 1

        # We finally normalize and return the score
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
        rspec_cu: cuda.devicearray,
        qspec_cu: cuda.devicearray,
        lens_cu: cuda.devicearray,
        norms_cu: cuda.device_array,
        out_cu: cuda.devicearray,
        stream: cuda.stream = None,
    ) -> None:
        """
        Launches the CUDA kernel for calculating cosine similarity scores.

        Parameters:
        -----------
        rspec_cu : cuda.devicearray
            Array containing reference spectra data.
        qspec_cu : cuda.devicearray
            Array containing query spectra data.
        lens_cu : cuda.devicearray
            Array containing lengths of spectra.
        out_cu : cuda.devicearray
            Array for storing similarity scores.
        out_cu : cuda.devicearray
            Precalculated norms for each R and Q
        stream : cuda.stream, optional
            CUDA stream for asynchronous execution, by default None.
        """
        _kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK, stream](
            rspec_cu,
            qspec_cu,
            lens_cu,
            norms_cu,
            out_cu,
        )
    return kernel

def modified_cosine_kernel(
    tolerance: float = 0.1,
    mz_power: float = 0.0,
    int_power: float = 1.0,
    match_limit: int = 1024,
    batch_size: int = 2048,
    n_max_peaks: int = 2048,
    is_symmetric: bool = False,
) -> callable:
    """
    Compiles and returns a CUDA kernel function for calculating cosine similarity scores between spectra.

    Parameters:
    -----------
    tolerance : float, optional
        Tolerance parameter for m/z matching, by default 0.1.
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

    Returns:
    --------
    callable
        CUDA kernel function for calculating cosine similarity scores.
    """

    if is_symmetric:
        warnings.warn("no effect from is_symmetric, it is not yet implemented")

    MATCH_LIMIT = match_limit * 2 # Since we now need twice as much space for both...
    N_MAX_PEAKS = n_max_peaks
    R, Q = batch_size, batch_size
    THREADS_PER_BLOCK_X = 1
    THREADS_PER_BLOCK_Y = 512
    THREADS_PER_BLOCK = (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)
    BLOCKS_PER_GRID_X = (R + THREADS_PER_BLOCK_X - 1) // THREADS_PER_BLOCK_X
    BLOCKS_PER_GRID_Y = (Q + THREADS_PER_BLOCK_Y - 1) // THREADS_PER_BLOCK_Y
    BLOCKS_PER_GRID = (BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y)

    @cuda.jit('void(float32[:,:,:], float32[:,:,:], int32[:,:], float32[:,:], float32[:,:,:])')
    def _kernel(
        rspec,
        qspec,
        lens,
        meta,
        out,
    ):
        """
        CUDA kernel function that will be translated to GPU-executable machine code on the fly.

        Parameters:
        -----------
        rspec : cuda.devicearray
            Array containing reference spectra data. Shape [2, R, M]
        qspec : cuda.devicearray
            Array containing query spectra data. Shape [2, Q, N]
        lens : cuda.devicearray
            Array containing lengths of spectra. Shape [2, Bs]
        meta : cuda.devicearray
            Metadata for each array. In this case, contains cosine norm and precursor_mz for each entry, shape [4, Bs], float.
        out : cuda.devicearray
            Array for storing similarity scores, [3, Bs, Bs], float.

        Notes:
        ------
        The kernel is designed to efficiently compute cosine similarity scores
        between reference and query spectra using CUDA parallelization.

        It performs the following steps:
        1. Find potential peak matches between the spectra based on m/z tolerance.
        2. Sort matches based on cosine product value.
        3. Accumulate cosine score, while discarding duplicate peaks.
        """
        ## PREAMBLE:
        # Get global indices
        i, j = cuda.grid(2)
        thread_i = cuda.threadIdx.x
        thread_j = cuda.threadIdx.y
        block_size_x = cuda.blockDim.x
        block_size_y = cuda.blockDim.y
        
        # Check we aren't out of the max possible grid size
        if i >= R or j >= Q:
            return

        # Set zeros, since we know we are in the grid
        out[0, i, j] = 0
        out[1, i, j] = 0
        out[2, i, j] = 0

        # Get actual length of R
        rleni = lens[0, i]
        qlenj = lens[1, j]
        
        if rleni == 0 or qlenj == 0:
            return
        
        # Norms are pre-calculated. 
        # reference norm is meta[0, :], query is meta[1, :]
        score_norm = meta[0, i] * meta[1, j]
        
        # reference precursor_mz is meta[2, :], query is meta[3, :]
        mass_shift = meta[2, i] - meta[3, j]
        
        # We unpack mz and int from arrays
        rmz = rspec[0] 
        rint = rspec[1]
        spec1_mz = rmz[i]
        spec1_int = rint[i]

        qmz = qspec[0]
        qint = qspec[1]
        spec2_mz = qmz[j]
        spec2_int = qint[j]

        spec1_mz_sh = spec1_mz # `spec1_mz_sh` is named so because at some point I experimented with R residing in shared-mem. It didn't work too well.
        spec1_int_sh = spec1_int

        #### PART 1: Find matches ####
        # On GPU global memory, we allocate temporary arrays for values and matches
        matches = cuda.local.array(MATCH_LIMIT, types.int32)
        values = cuda.local.array(MATCH_LIMIT, types.float32)
        # TODO
        ## Different ways to implement this
        # ONE:
        # Allocate 2x ML
        # Run first fully, second fully
        # How to sort? We have bunch of empty space in between. We need third fgin array to copy both to, to sort correctly (or pad with zeros)
        # Optionally, prefill val arr with zeros, and then sort it. 
        # SECOND:
        # Allocate 2x ML
        # Run fist (add), immediately add second (shifted version) too,
        # Same loop iter
        ## 
        # This will give possibly interleaved pattern of matches i.e. [match, sft-match, match, match, sft, sft, sft, ... ]
        # 
        
        num_match = 0

        lowest_idx = 0
        shift = 0
        overflow = False
        for peak1_idx in range(rleni):
            if overflow:
                break
            mz_r = spec1_mz_sh[peak1_idx]
            int_r = spec1_int_sh[peak1_idx]
            for peak2_idx in range(lowest_idx, qlenj):
                if overflow:
                    break
                mz_q = spec2_mz[peak2_idx]
                if mz_q + shift > mz_r + tolerance:
                    break
                if mz_q + shift + tolerance < mz_r:
                    lowest_idx = peak2_idx + 1
                else:
                    if not overflow:
                        int_q = spec2_int[peak2_idx]
                        # Binary trick! We pack two 16bit ints in 32bit int to use less memory
                        # since we know that largest imaginable peak index can fit in 13 bits
                        matches[num_match] = (peak1_idx << 16) | peak2_idx
                        values[num_match] = (mz_r ** mz_power * int_r ** int_power) * (mz_q ** mz_power * int_q ** int_power)
                        num_match += 1
                        overflow = num_match >= (MATCH_LIMIT//2)  # This is the errorcode for overflow
        
        shift = mass_shift
        lowest_idx = 0
        overflow_ms = False
        for peak1_idx in range(rleni):
            if overflow_ms:
                break
            mz_r = spec1_mz_sh[peak1_idx]
            int_r = spec1_int_sh[peak1_idx]
            for peak2_idx in range(lowest_idx, qlenj):
                if overflow_ms:
                    break
                mz_q = spec2_mz[peak2_idx]
                if mz_q + shift > mz_r + tolerance:
                    break
                if mz_q + shift + tolerance < mz_r:
                    lowest_idx = peak2_idx + 1
                else:
                    if not overflow_ms:
                        int_q = spec2_int[peak2_idx]
                        # Binary trick! We pack two 16bit ints in 32bit int to use less memory
                        # since we know that largest imaginable peak index can fit in 13 bits
                        matches[num_match] = (peak1_idx << 16) | peak2_idx
                        values[num_match] = (mz_r ** mz_power * int_r ** int_power) * (mz_q ** mz_power * int_q ** int_power)
                        num_match += 1
                        overflow_ms = num_match >= MATCH_LIMIT  # This is the errorcode for overflow

        out[2, i, j] = overflow + 2 * overflow_ms
        
        ## Second part here...

        if num_match == 0:
            return
        # Debug checkpoint
        # out[i, j, 0] = score_norm
        # out[i, j, 1] = num_match
        # return

        #### PART: 2 ####
        # We use as non-recursive mergesort to order matches by the cosine product
        # We need an O(MATCH_LIMIT) auxiliary memory for this.

        temp_matches = cuda.local.array(MATCH_LIMIT, types.int32)
        temp_values = cuda.local.array(MATCH_LIMIT, types.float32)

        k = types.int32(1)
        while k < num_match:
            for left in range(0, num_match - k, k * 2):
                rght = left + k
                rend = rght + k
                
                rend = min(rend, num_match)

                m = left; ix = left; jx = rght;
                while ix < rght and jx < rend:
                    mask = (values[ix] > values[jx])
                    temp_matches[m] = mask * matches[ix] + (1 - mask) * matches[jx]
                    temp_values[m] = mask * values[ix] + (1 - mask) * values[jx]
                    ix+=mask
                    jx+=(1-mask)
                    m+=1

                while ix < rght:
                    temp_matches[m] = matches[ix]; 
                    temp_values[m] = values[ix]; 
                    ix+=1; m+=1;
                
                while jx < rend:
                    temp_matches[m] = matches[jx]; 
                    temp_values[m] = values[jx]; 
                    jx+=1; m+=1;
                
                for m in range(left, rend):
                    matches[m] = temp_matches[m]; 
                    values[m] = temp_values[m]; 
            k *= 2

        #### PART: 3 ####
        # Accumulate and deduplicate matches from largest to smallest ####
        used_r = cuda.local.array(N_MAX_PEAKS, types.boolean)
        used_q = cuda.local.array(N_MAX_PEAKS, types.boolean)
        for m in range(N_MAX_PEAKS):
            used_r[m] = False
            used_q[m] = False

        used_matches = 0
        score = 0.0
        for m in range(num_match):
            # Here we undo the binary trick
            c = matches[m]
            peak1_idx = c >> 16
            peak2_idx = c & 0x0000_FFFF
            if (not used_r[peak1_idx]) and (not used_q[peak2_idx]):
                used_r[peak1_idx] = True
                used_q[peak2_idx] = True
                score += values[m];
                used_matches += 1

        #### ALTERNATIVE PART 2-3: ALTENRNATIVE SORT+ACCUMULATE PATHWAY ####
        # This pathway is much faster when matches and average scores are *extremely* rare
        # TODO: 
        # In the future, we should compile both kernels, compare perfs and use fastest kernel.
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

        out[0, i, j] = score / score_norm
        out[1, i, j] = used_matches

    def kernel(
        rspec_cu: cuda.devicearray,
        qspec_cu: cuda.devicearray,
        lens_cu: cuda.devicearray,
        norms_cu: cuda.device_array,
        out_cu: cuda.devicearray,
        stream: cuda.stream = None,
    ) -> None:
        """
        Launches the CUDA kernel for calculating cosine similarity scores.

        Parameters:
        -----------
        rspec_cu : cuda.devicearray
            Array containing reference spectra data.
        qspec_cu : cuda.devicearray
            Array containing query spectra data.
        lens_cu : cuda.devicearray
            Array containing lengths of spectra.
        out_cu : cuda.devicearray
            Array for storing similarity scores.
        out_cu : cuda.devicearray
            Precalculated norms for each R and Q
        stream : cuda.stream, optional
            CUDA stream for asynchronous execution, by default None.
        """
        _kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK, stream](
            rspec_cu,
            qspec_cu,
            lens_cu,
            norms_cu,
            out_cu,
        )
    return kernel


def cpu_parallel_cosine_greedy(
    tolerance: float = 0.1,
    shift: float = 0,
    mz_power: float = 0.0,
    int_power: float = 1.0,
    match_limit: int = 1024,
    batch_size: int = 2048,
    n_max_peaks: int = 2048,
    is_symmetric: bool = False,
) -> callable:
    pass    