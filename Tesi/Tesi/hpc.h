/****************************************************************************
 *
 * hpc.h - Miscellaneous utility functions for the HPC course adapted for this program needs.
 *
 * Written in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Last modified in 2021 by Andrea Ingargiola
 *
 * To the extent possible under law, the author(s) have dedicated all
 * copyright and related and neighboring rights to this software to the
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see
 * <http://creativecommons.org/publicdomain/zero/1.0/>.
 *
 * --------------------------------------------------------------------------
 *
 * This header file provides a function double hpc_gettime() that
 * returns the elapsed time (in seconds) since "the epoch". The
 * function uses the timing routing of the underlying parallel
 * framework (OpenMP or MPI), if enabled; otherwise, the default is to
 * use the clock_gettime() function.
 *
 * IMPORTANT NOTE: to work reliably this header file must be the FIRST
 * header file that appears in your code.
 *
 ****************************************************************************/

#ifndef HPC_H
#define HPC_H

#ifdef __CUDACC__

#include <stdio.h>
#include <stdlib.h>

/* from https://gist.github.com/ashwin/2652488 */

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
#ifndef NO_CUDA_CHECK_ERROR
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));
        abort();
    }
#endif
}

inline void __cudaCheckError(const char* file, const int line)
{
#ifndef NO_CUDA_CHECK_ERROR
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));
        abort();
    }

    /* More careful checking. However, this will affect performance.
       Comment away if needed. */
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));
        abort();
    }
#endif
}

#endif

#endif

