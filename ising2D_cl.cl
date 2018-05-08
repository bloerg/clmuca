#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable


// 256 threads per block ensures the possibility of full occupancy
// for all compute capabilities if thread count small enough
#define WORKERS_PER_BLOCK 256
#define WORKER get_global_id(0)


// Random Number Generator
#include "philox.h"
#include "u01fixedpt.h"


typedef unsigned long long my_uint64;

// calculate bin index from energy E
inline ulong EBIN(int E, int* d_N)
{
  return (E + (*d_N << 1)) >> 2;
}

// calculate energy difference of one spin flip
inline int localE(uint idx, __global char* lattice, int* d_L, int* d_N, int* d_NUM_WORKERS)
{
  int right = idx + 1;
  int left = convert_int( idx ) -1; 

  int up = idx + *d_L;
  int down = convert_int( idx ) - *d_L;

  // check periodic boundary conditions
  if (right % *d_L == 0) right -= *d_L;
  if (idx % *d_L == 0) left += *d_L;
  if (up > convert_int(*d_N - 1) ) up -= *d_N;
  if (down < 0 ) down += *d_N;

  return -lattice[idx * *d_NUM_WORKERS + WORKER] *
     ( lattice[right * *d_NUM_WORKERS + WORKER] +
       lattice[left * *d_NUM_WORKERS + WORKER] +
       lattice[up * *d_NUM_WORKERS + WORKER] + 
       lattice[down * *d_NUM_WORKERS + WORKER] 
     );
}

// calculate total energy
int calculateEnergy(__global char* lattice, int* d_L, int* d_N, int* d_NUM_WORKERS)
{
  int sum = 0;

  for (size_t i = 0; i < *d_N; i++) {
    sum += localE(i, lattice, d_L, d_N, d_NUM_WORKERS);
  }
  // divide out double counting
  return (sum >> 1); 
}


// multicanonical Markov chain update (single spin flip)
inline bool mucaUpdate(float rannum, int* energy, __global char* d_lattice, __global float* d_log_weights, uint idx, int* d_L, int* d_N, int* d_NUM_WORKERS)
{
  // precalculate energy difference
  int dE = -2 * localE(idx, d_lattice, d_L, d_N, d_NUM_WORKERS);

  // flip with propability W(E_new)/W(E_old)
  // weights are stored in texture memory for faster random access

  //FIXME: tex1Dfetch equivalent in OPENCL? maybe images
  //~ if (rannum < expf(tex1Dfetch(t_log_weights, EBIN(*energy + dE)) - tex1Dfetch(t_log_weights, EBIN(*energy)))) {
  if (rannum < exp(d_log_weights[EBIN(*energy + dE, d_N)] - d_log_weights[EBIN(*energy, d_N)] ) ) {
    d_lattice[idx * *d_NUM_WORKERS + WORKER] = -d_lattice[idx * *d_NUM_WORKERS + WORKER];
    *energy += dE;
    return true;
  }
  return false;
}


__kernel void computeEnergies(__global char* d_lattice, __global int* d_energies, __private int d_L, __private int d_N, __private int d_NUM_WORKERS)
{
  d_energies[WORKER] = calculateEnergy(d_lattice, &d_L, &d_N, &d_NUM_WORKERS);
  barrier(CLK_GLOBAL_MEM_FENCE);

}


__kernel void mucaIteration(
  __global char* d_lattice, 
  __global ulong* d_histogram, 
  __global int* d_energies, 
  __global float* d_log_weights,
  __private ulong iteration, 
  __private uint seed, 
  __private ulong d_NUPDATES_THERM, 
  __private ulong d_NUPDATES,
  __private int d_L, 
  __private int d_N, 
  __private int d_NUM_WORKERS
)
{
  // initialize two RNGs
  // one for acceptance propability (k1)
  // and one for selection of a spin (same for all workers) (k2)

  philox4x32_key_t k1 = {{WORKER, 0xdecafbad}};
  philox4x32_key_t k2 = {{0xC001CAFE, 0xdecafbad}};
  philox4x32_ctr_t c = {{0, seed, iteration, 0xBADC0DED}};//0xBADCAB1E
  philox4x32_ctr_t r1, r2;
 
  // reset global histogram
  for (size_t i = 0; i < ((d_N + 1) / d_NUM_WORKERS) + 1; i++) {
    if (i*d_NUM_WORKERS + WORKER < d_N + 1) {
      d_histogram[i * d_NUM_WORKERS + WORKER] = 0;
    }
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  int energy;
  energy = d_energies[WORKER];

  // thermalization
  for (size_t i = 0; i < d_NUPDATES_THERM; i++) {
    if(i%4 == 0) {
      ++c.v[0];
      //r1 = rng(c, k1); r2 = rng(c, k2);
      r1 = philox4x32_R(7, c, k1); r2 = philox4x32_R(7, c, k2); //7 rounds
    }
    uint idx = convert_uint(u01fixedpt_open_open_32_24(r2.v[i%4]) * d_N); // 32_24 = float;  64_53 = double
    mucaUpdate(u01fixedpt_open_open_32_24(r1.v[i%4]), &energy, d_lattice, d_log_weights, idx, &d_L, &d_N, &d_NUM_WORKERS);
  }
  barrier(CLK_LOCAL_MEM_FENCE);


  // estimate current propability distribution of W(E)
  for (ulong i = 0; i < d_NUPDATES; i++) {
    if(i%4 == 0) {
      ++c.v[0];
      r1 = philox4x32_R(7, c, k1); r2 = philox4x32_R(7, c, k2);
    }
    uint idx = convert_uint(u01fixedpt_open_open_32_24(r2.v[i%4]) * d_N); // 24_32 = float;  64_53 = double
    mucaUpdate(u01fixedpt_open_open_32_24(r1.v[i%4]), &energy, d_lattice, d_log_weights, idx, &d_L, &d_N, &d_NUM_WORKERS);
    // add to global histogram
    //~ d_histogram[EBIN(energy, &d_N)] += 1; // incorrect results because non-atomic
    //~ atomic_add(d_histogram + EBIN(energy, &d_N), 1); //Problem: this works only with 32 bit types in OpenCL, which is too small
    atom_add(d_histogram + EBIN(energy, &d_N), 1); //Problem: this works with 64Bit but requires support of cl_khr_int64_base_atomics pragma
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  d_energies[WORKER] = energy;
}
