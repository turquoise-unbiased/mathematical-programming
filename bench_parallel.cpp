/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms and short vectors using double precision floating-point values
   [icpx 2025.0.1]
*/
#include <stdio.h>
#include <float.h>
#include <fenv.h>
#include <math.h>

#include <oneapi/tbb/flow_graph.h>
#include <oneapi/tbb/parallel_for_each.h>
#include <oneapi/tbb/parallel_sort.h>

#define SVX (2L << 4L)  // short vector elements 2^n [>= 2]

#include "bench_tpl.hpp"  // integral benchmark templates

static_assert(((SVX >= 2L) and not (SVX&(SVX-1L))));

typedef double svxdf_t __attribute__ ((vector_size ((SVX * sizeof(double)))));  // short vector arithmetic type
static_assert((sizeof(svxdf_t) == alignof(svxdf_t)));  // reduction

constexpr double MILLE = 1e6;  // trial_scale [>= SVX]
static_assert((MILLE >= SVX));

// trial namespace
namespace trial {
  const int STATUS_OK = (SVRNG_STATUS_OK | TBBMALLOC_OK);  // constructors ok  [bench_tpl::RNG|vector]
  enum class FUSION { ON = 1, OFF };  // loop fusion constants
  size_t fn(const size_t N);
  size_t (*fnptr)(size_t) = fn;  // fn pointer
}  // end trial

size_t trial::fn(const size_t N) {
  const FUSION fusion = FUSION::OFF;  // loop fusion constant
  const double trial_scale = scalbln(MILLE, N);  // trial_scale
  const size_t trial_size  = lrint(trial_scale);  // trial_size
  // diagnostic
  tpl::bench bench;
  // random number generator
  const tpl::RNG rng(trial_scale);
  using svrngx_t = decltype(rng)::svrngx_t;
  int rng_st;  // RNG status
  // container vector, pointers, reducers
  const tpl::vector<double> vec(trial_size);
  double v_mean, v_median, v_mad, v_sum = 0e0;  // stats
  // double for each
  switch ( rng_st = svrng_get_status(); ( rng_st | vec.st ) ) {
    case STATUS_OK:
      // svrng pointers
      svrngx_t* const rd_begin = (svrngx_t*)(&(*vec.begin));
      const svrngx_t* const rd_end = vec.sv_mod<svrngx_t>();  // SVRNG modulus
      double* const rd_rem = vec.v_mod<svrngx_t>();  // vector modulus
      // short vector pointers
      const svxdf_t* const sv_begin = (svxdf_t*)(&(*vec.begin));
      const svxdf_t* const sv_end = vec.sv_mod<svxdf_t>();  // short vector modulus
      const double* const sv_rem = vec.v_mod<svxdf_t>();  // vector modulus
      // reducers
      svxdf_t sv_acc = { 0e0 };  // short vector reducer
      const double* const sv_0 = (double*)(&sv_acc);  // short vector reducer pointers
      const double* const sv_f = (sv_0 + (sizeof(svxdf_t) / sizeof(double)));
      // FUSION
      switch (fusion) {
        case FUSION::ON:
          bench.push(tick_count::now());  // 1&2)
          for(double* k = vec.begin; k < vec.end; ++k) { v_sum += ( *k = rng.unif() ); }
          bench.push(tick_count::now());
        break;
        case FUSION::OFF:
          bench.push(tick_count::now());  // 1)
          for(svrngx_t* k = rd_begin; k < rd_end; ++k) { *k = rng.unifsvx(); }
          for(double* k = rd_rem; k < vec.end; ++k) { *k = rng.unif(); }  // remainder
          bench.push(tick_count::now());

          if(( rng_st = svrng_get_status() ) != SVRNG_STATUS_OK) { break; }

          // sum
          bench.push(tick_count::now());  // 2)
          for(const svxdf_t* k = sv_begin; k < sv_end; ++k) { sv_acc += *k; }
          for(const double* k = sv_0; k < sv_f; ++k) { v_sum += *k; }
          for(const double* k = sv_rem; k < vec.end; ++k) { v_sum += *k; }  // remainder
          bench.push(tick_count::now());
        break;
      }  // FUSION
    break;
  }  // STATUS
  // mean, median, mad
  // computing or jump according with RNG status
  switch ( rng_st = svrng_get_status(); ( rng_st | vec.st ) ) {
    case STATUS_OK:
      v_mean = tpl::mean(v_sum, vec.size);

      bench.push(tick_count::now());  // 3)
      parallel_sort(vec.begin, vec.end);
      bench.push(tick_count::now());

      v_median = tpl::median(vec.begin, vec.end, vec.size);

      bench.push(tick_count::now());  // 4)
      parallel_for_each(vec.begin, vec.end, [&](auto &elem) { elem = fabs((elem - v_median)); });
      bench.push(tick_count::now());

      bench.push(tick_count::now());  // 5)
      parallel_sort(vec.begin, vec.end);
      bench.push(tick_count::now());

      v_mad = tpl::median(vec.begin, vec.end, vec.size);

      // print stats
      printf( "%zu) trial size:                        %zu doubles\n", N, vec.size           );
    switch (fusion) { case FUSION::ON:
      printf( "1&2) for generate [fused reduce]      %.6fs\n", bench.f1()                    );
    break; case FUSION::OFF:
      printf( "1) for generate                       %.6fs\n", bench.f1()                    );
      printf( "2) for reduce                         %.6fs\n", bench.f1()                    );
    break; }  // FUSION
      printf( "3) parallel_sort                      %.6fs\n", bench.f1()                    );
      printf( "4) parallel_for_each                  %.6fs\n", bench.f1()                    );
      printf( "5) parallel_sort                      %.6fs\n", bench.f1()                    );
      printf( "1) sum: sum(v)                        %.23e\n", v_sum                         );
      printf( "2) mean: sum/size(v)                  %.17e\n", v_mean                        );
      printf( "3) median: sort(v)[med]               %.17e\n", v_median                      );
      printf( "4) mad: sort(v-median)[med]           %.17e\n", v_mad                         );
      // implementation-dependent arithmetic types
      printf( "a) Machine epsilon (f):               %e\n",  FLT_EPSILON                     );
      printf( "b) Machine epsilon (ff):              %e\n",  DBL_EPSILON                     );
      printf( "c) Machine epsilon (fff):             %Le\n", LDBL_EPSILON                    );
      printf( "d) Machine rounds style:              %i\n",  FLT_ROUNDS                      );
      printf( "e) x87 FPU exception flags:           %i\n\n", fetestexcept(FE_ALL_EXCEPT)    );
    break;
    default:
      printf( "APP FAILED: status rng: %i | tbb: %i\n", rng_st, vec.st );
    return 0L;
  }  // STATUS
  return trial_size;
}  // trial::fn

int main() {
  size_t test_size = 0L;  // compound parallel size
  // graph parallelism
  flow::graph graph;
  // function_node
  flow::function_node<size_t, size_t, flow::queueing> fn(graph, flow::unlimited,  // flow::serial
    [&](const size_t &n) { return trial::fnptr(n); });
  // function_node
  flow::function_node<size_t, size_t, flow::queueing_lightweight> comp(graph, flow::serial,
    [&](const size_t &z) { return __atomic_add_fetch(&test_size, z, __ATOMIC_SEQ_CST); });
  // edge
  flow::make_edge(fn, comp);
  // diagnostic
  tpl::bench bench;
  bench.push(tick_count::now());  // c)
  // execute incoming message
  for(size_t n = 0L; n < 3L; ++n) { fn.try_put(n); }
  // block & catch & throw
  try { graph.wait_for_all(); }
  catch (...) { graph.cancel(); throw; }
  bench.push(tick_count::now());
  // print info
  printf( "a) graph exception thrown:            %i\n", graph.exception_thrown()   );
  printf( "b) compound test size:                %zu\n", test_size                 );
  printf( "c) compound test time:                %.6fs\n", bench.f1()              );

  return 0;
}
