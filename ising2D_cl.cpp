// OpenCL
#include <CL/cl.hpp>

// This includes my_uint64 type
#include "ising2D_io.hpp"
#include "muca.hpp"

#include <sys/time.h>

// Random Number Generator
#include "Random123/philox.h"
#include "Random123/examples/uniform.hpp"

// 256 threads per block ensures the possibility of full occupancy
// for all compute capabilities if thread count small enough
#define WORKERS_PER_BLOCK 256

// choose random number generator
typedef r123::Philox4x32_R<7> RNG;

using namespace std;

int main(int argc, char** argv) 
{
  parseArgs(argc, argv);
  if (NUM_WORKERS % WORKERS_PER_BLOCK != 0) {
    cerr << "ERROR: NUM_WORKERS must be multiple of " << WORKERS_PER_BLOCK << endl;
  }

  // select device
  vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.size() == 0) {
      cerr << "ERROR: No platforms found. Check OpenCL installation!\n";
      exit(1);
  }
  cl::Platform default_platform=all_platforms[1];
  cout << "\nINFO: Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

  vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices); 
  int deviceCount = all_devices.size();
  if (deviceCount == 0) {
      cerr << "ERROR: No devices found. Check OpenCL installation!\n";
      exit(1);
  }

  cl::Device device;
  if(REQUESTED_GPU >= 0 and REQUESTED_GPU < deviceCount)
  {
    device = all_devices[REQUESTED_GPU];
  }
  else 
  {
    device = all_devices[0];
  }
  cout << "INFO: Using device " << device.getInfo<CL_DEVICE_NAME>() << "\n";

  // figure out optimal execution configuration
  // based on GPU architecture and generation

  int maxresidentthreads = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  int totalmultiprocessors = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  int optimum_number_of_workers = maxresidentthreads*totalmultiprocessors;
  if (NUM_WORKERS == 0) {
    NUM_WORKERS = optimum_number_of_workers * 2;
  }
  cout << "INFO: Number of Workers: " << NUM_WORKERS << "\n";

  cout << "INFO: GPU capabilities\n      CL_DEVICE_MAX_WORK_GROUP_SIZE: " << maxresidentthreads << "\n      CL_DEVICE_MAX_COMPUTE_UNITS: " << totalmultiprocessors << "\n";
 
  cl::Context cl_context({device});
  cl::CommandQueue cl_queue(cl_context, device);

 // read the kernel from source file
  std::ifstream cl_program_file_ising("ising2D_cl.cl");
  std::string cl_program_string_ising(
    std::istreambuf_iterator<char>(cl_program_file_ising),
    (std::istreambuf_iterator<char>())
  );

  cl::Program cl_program_ising(cl_context, cl_program_string_ising, true);

  if (cl_program_ising.build({ device }, "-I include") != CL_SUCCESS){
    cerr << "ERROR: Error building: " << cl_program_ising.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
    getchar();
    exit(1);
  }

  cl::Kernel cl_kernel_compute_energies(cl_program_ising, "computeEnergies");
  cl::Kernel cl_kernel_muca_iteration(cl_program_ising, "mucaIteration");

  int memory_operation_status; //for debugging
  int kernel_run_result; //for debugging

  // initialize NUM_WORKERS (LxL) lattices
  RNG rng;
  vector<cl_char> h_lattice(NUM_WORKERS * N);

  cl::Buffer d_lattice_buf (
    cl_context,
    CL_MEM_READ_WRITE,
    NUM_WORKERS * N * sizeof(cl_char),
    NULL,
    &memory_operation_status
  );

  for (unsigned worker=0; worker < NUM_WORKERS; worker++) {
    RNG::key_type k = {{worker, 0xdecafbad}};
    RNG::ctr_type c = {{0, seed, 0xBADCAB1E, 0xBADC0DED}};
    RNG::ctr_type r;
    for (size_t i = 0; i < N; i++) {
      if (i%4 == 0) {
        ++c[0];
        r = rng(c, k);
      }
      h_lattice.at(i * NUM_WORKERS + worker) = 2 * (r123::u01fixedpt<float>(r.v[i%4]) < 0.5) - 1;
    }
  }
 
//  cout << "DEBUG: return value of create buffer d_lattice_buf: " << memory_operation_status << "\n";
  memory_operation_status = cl_queue.enqueueWriteBuffer(d_lattice_buf, CL_TRUE, 0, NUM_WORKERS * N * sizeof(cl_char), &h_lattice[0]);
//  cout << "DEBUG: return value of writing d_lattice_buf to device" << memory_operation_status << "\n";


  // initialize all energies
  cl::Buffer d_energies_buf (
    cl_context,
    CL_MEM_READ_WRITE,
    NUM_WORKERS * sizeof(cl_int),
    NULL,
    &memory_operation_status
  );
//  cout << "DEBUG: return value of create buffer d_energies_buf: " << memory_operation_status << "\n";

  cl_kernel_compute_energies.setArg(0, d_lattice_buf);
  cl_kernel_compute_energies.setArg(1, d_energies_buf);
  cl_kernel_compute_energies.setArg(2, L);
  cl_kernel_compute_energies.setArg(3, N);
  cl_kernel_compute_energies.setArg(4, NUM_WORKERS);

  kernel_run_result = cl_queue.enqueueNDRangeKernel(
    cl_kernel_compute_energies, 
    cl::NDRange(0), 
    cl::NDRange(NUM_WORKERS), 
    cl::NDRange(WORKERS_PER_BLOCK)
  );
//  cout << "DEBUG: return value of cl_kernel_compute_energies start: " << kernel_run_result << "\n";


  // initialize ONE global weight array
  vector<cl_float> h_log_weights(N + 1, 0.0f); 

  cl::Buffer d_log_weights_buf (
    cl_context,
    CL_MEM_READ_WRITE,
    ( N + 1 ) * sizeof(cl_float),
    NULL,
    &memory_operation_status
  );
//  std::cout << "DEBUG: return value of create buffer d_log_weights: " << memory_operation_status << "\n";

  //~ FIXME: I there a way to do the following in Opencl? Maybe images.
  //~ cudaBindTexture(NULL, t_log_weights, d_log_weights, (N + 1) * sizeof(float));

  // initialize ONE global histogram
  vector<my_uint64> h_histogram((N + 1), 0); 
  cl::Buffer d_histogram_buf (
    cl_context,
    CL_MEM_READ_WRITE,
    ( N + 1 ) * sizeof(my_uint64),
    NULL,
    &memory_operation_status
  );
//  cout << "DEBUG: return value of create buffer d_histogram_buf:: " << memory_operation_status << "\n";
  memory_operation_status = cl_queue.enqueueWriteBuffer(d_histogram_buf, CL_TRUE, 0, (N+1) * sizeof(my_uint64), &h_histogram[0]);
//  cout << "DEBUG: return value of writing d_histogram_buf to device: " << memory_operation_status << "\n";


  // timing and statistics
  vector<long double> times;
  timespec start, stop;
  ofstream iterfile;
  
  iterfile.open("run_iterations.dat");
  // initial estimate of width at infinite temperature 
  // (random initialization requires practically no thermalization)
  unsigned width = 10;
  double nupdates_run = 1;
  // heuristic factor that determines the number of statistic per iteration
  // should be related to the integrated autocorrelation time
  double z = 2.25;


  // main iteration loop
  for (cl_ulong k=0; k < MAX_ITER; k++) {
    cout << "DEBUG: Starting iteration " << k << "\n";
    // start timer
    //~ cudaDeviceSynchronize(); //This should be true in OpenCL after the last kernel run and buffer read.
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    // copy global weights to GPU
    memory_operation_status = cl_queue.enqueueWriteBuffer(d_log_weights_buf, CL_TRUE, 0, (N+1) * sizeof(cl_float), &h_log_weights[0]);
    cout << "DEBUG: return value of writing d_log_weights_buf to device: " << memory_operation_status << "\n";

    // acceptance rate and correlation time corrected "random walk"
    // in factor 30 we adjusted acceptance rate and >L range requirement of our present Ising situation
    NUPDATES_THERM = 30 * width;
    if (width<N) {
      // 6 is motivated by the average acceptance rate of a multicanonical simulation ~0.45 -> (1/0.45)**z~6
      nupdates_run = 6 * pow(width, z) / NUM_WORKERS;
    }
    else {
      // for a flat spanning histogram, we assume roughly equally distributed
      // walkers and reduce the thermalization time
      // heuristic modification factor;
      // (>1; small enough to introduce statistical fluctuations on the convergence measure)
      nupdates_run *= 1.1;
    }
    NUPDATES = static_cast<my_uint64>(nupdates_run)+1;
    // local iteration on each thread, writing to global histogram
    
    cl_kernel_muca_iteration.setArg(0, d_lattice_buf);
    cl_kernel_muca_iteration.setArg(1, d_histogram_buf);
    cl_kernel_muca_iteration.setArg(2, d_energies_buf);
    cl_kernel_muca_iteration.setArg(3, d_log_weights_buf);
    cl_kernel_muca_iteration.setArg(4, k);
    cl_kernel_muca_iteration.setArg(5, seed);
    cl_kernel_muca_iteration.setArg(6, NUPDATES_THERM);
    cl_kernel_muca_iteration.setArg(7, NUPDATES);
    cl_kernel_muca_iteration.setArg(8, L);
    cl_kernel_muca_iteration.setArg(9, N);
    cl_kernel_muca_iteration.setArg(10, NUM_WORKERS);

    kernel_run_result = cl_queue.enqueueNDRangeKernel(cl_kernel_muca_iteration, cl::NDRange(0), cl::NDRange(NUM_WORKERS), cl::NDRange(WORKERS_PER_BLOCK));
//    cout << "DEBUG: Result of cl_kernel_muca_iteration start: " << kernel_run_result << "\n";


    // copy global histogram back to CPU
    memory_operation_status = cl_queue.enqueueReadBuffer(d_histogram_buf, CL_TRUE, 0, ( N + 1 ) * sizeof(my_uint64), &h_histogram[0]);
//    cout << "DEBUG: return value of reading d_histogram_buf from device: " << memory_operation_status << "\n";


    // stop timer
    //~ cudaDeviceSynchronize();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    long double elapsed = 1e9* (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);
    times.push_back(elapsed);
    TOTAL_THERM   += NUPDATES_THERM;
    TOTAL_UPDATES += NUPDATES;

    // flatness in terms of kullback-leibler-divergence; 
    // requires sufficient thermalization!!! 
    double dk  = d_kullback(h_histogram);
    iterfile << "#NITER = " << k  << " dk=" << dk << endl;
    writeHistograms(h_log_weights, h_histogram, iterfile);
    if (dk<1e-4) {
      break;
    }

    // measure width of the current histogram
    size_t start,end;
    getHistogramRange(h_histogram, start, end);
    unsigned width_new = end-start;
    if (width_new > width) width=width_new;
    // update logarithmic weights with basic scheme if not converged
    updateWeights(h_log_weights, h_histogram);
  }
  iterfile.close();
  ofstream sout;
  sout.open("stats.dat");
  writeStatistics(times, sout);
  sout << "total number of thermalization steps/Worker : " << TOTAL_THERM << "\n";
  sout << "total number of iteration updates   /Worker : " << TOTAL_UPDATES << "\n";
  sout << "total number of all updates         /Worker : " << TOTAL_THERM+TOTAL_UPDATES << "\n";
  sout.close();
 
  if (production) {
    std::cout << "start production run ..." << std::endl;
    // copy global weights to GPU
    memory_operation_status = cl_queue.enqueueWriteBuffer(d_log_weights_buf, CL_TRUE, 0, (N+1) * sizeof(cl_float), &h_log_weights[0]);
//    cout << "DEBUG: return value of writing d_log_weights_buf to device: " << memory_operation_status << "\n";

    // thermalization
    NUPDATES_THERM = pow(N,z);

    cl_kernel_muca_iteration.setArg(0, d_lattice_buf);
    cl_kernel_muca_iteration.setArg(1, d_histogram_buf);
    cl_kernel_muca_iteration.setArg(2, d_energies_buf);
    cl_kernel_muca_iteration.setArg(3, d_log_weights_buf);
    cl_kernel_muca_iteration.setArg(4, 0);
    cl_kernel_muca_iteration.setArg(5, seed+1000);
    cl_kernel_muca_iteration.setArg(6, NUPDATES_THERM);
    cl_kernel_muca_iteration.setArg(7, 0);
    cl_kernel_muca_iteration.setArg(8, L);
    cl_kernel_muca_iteration.setArg(9, N);
    cl_kernel_muca_iteration.setArg(10, NUM_WORKERS);

    kernel_run_result = cl_queue.enqueueNDRangeKernel(cl_kernel_muca_iteration, cl::NDRange(0), cl::NDRange(NUM_WORKERS), cl::NDRange(WORKERS_PER_BLOCK));
//    cout << "DEBUG: Result of cl_kernel_muca_iteration start: " << kernel_run_result << "\n";

    // set jackknife  
    size_t JACKS = 100;
    NUPDATES = NUPDATES_PRODUCTION/JACKS;
    // loop over Jackknife bins
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    for (size_t k = 0; k < JACKS; k++) {
      //~ cudaDeviceSynchronize();
      // local production on each thread, writing to global histogram
      cl_kernel_muca_iteration.setArg(0, d_lattice_buf);
      cl_kernel_muca_iteration.setArg(1, d_histogram_buf);
      cl_kernel_muca_iteration.setArg(2, d_energies_buf);
      cl_kernel_muca_iteration.setArg(3, d_log_weights_buf);
      cl_kernel_muca_iteration.setArg(4, k);
      cl_kernel_muca_iteration.setArg(5, seed+2000);
      cl_kernel_muca_iteration.setArg(6, 0);
      cl_kernel_muca_iteration.setArg(7, NUPDATES);
      cl_kernel_muca_iteration.setArg(8, L);
      cl_kernel_muca_iteration.setArg(9, N);
      cl_kernel_muca_iteration.setArg(10, NUM_WORKERS);

      kernel_run_result = cl_queue.enqueueNDRangeKernel(cl_kernel_muca_iteration, cl::NDRange(0), cl::NDRange(NUM_WORKERS), cl::NDRange(WORKERS_PER_BLOCK));
//      cout << "DEBUG: Result of cl_kernel_muca_iteration start: " << kernel_run_result << "\n";


      // copy global histogram back to CPU
      memory_operation_status = cl_queue.enqueueReadBuffer(d_histogram_buf, CL_TRUE, 0, ( N + 1 ) * sizeof(my_uint64), &h_histogram[0]);
//      cout << "DEBUG: return value of reading d_histogram_buf from device: " << memory_operation_status << "\n";

      std::stringstream filename;
      filename << "production" << std::setw(3) << std::setfill('0') << k << ".dat";
      iterfile.open(filename.str().c_str());
      writeHistograms(h_log_weights, h_histogram, iterfile);
      iterfile.close();
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    std::cout << "production run updates  JACK: " << NUPDATES     << "*WORKER \n";
    std::cout << "production run updates total: " << NUPDATES*100 << "*WORKER \n";
    std::cout << "production run time total   : " << (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)*1e-9 << "s\n"; 
    sout.open("stats.dat", std::fstream::out | std::fstream::app);
    sout << "production run updates  JACK: " << NUPDATES     << "*WORKER \n";
    sout << "production run updates total: " << NUPDATES*100 << "*WORKER \n";
    sout << "production run time total   : " << (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)*1e-9 << "s\n"; 
    sout.close();
  }

return 0;
}
