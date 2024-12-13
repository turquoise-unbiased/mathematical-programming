/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms c++17 & lambdas c++11 */
#include <float.h>
#include <mathimf.h>
#include <svrng.h>
#include <oneapi/tbb.h>
#include <oneapi/tbb/scalable_allocator.h>

#include <cstdio>
#include <vector>

constexpr auto N = 1'000'000L;
using namespace oneapi::tbb;

int fn( void ) {
  // diagnostic
  tick_count t0, t1;
  concurrent_queue<tick_count> t;
  auto f1 = [&]() { return ((int(t.try_pop(t0)) & int(t.try_pop(t1))) ? (t1-t0).seconds() : -(0e0)); };
  // random number generator
  svrng_engine_t engine      = svrng_new_rand_engine(37u);
  svrng_distribution_t distr = svrng_new_uniform_distribution_double(1e0, (N / 2e0));
  // container vector
  std::vector<double, scalable_allocator<double>> v((N | 1L));
  double v_sum = 0e0, v_mean = 0e0, v_median = 0e0, v_mad = 0e0;
  // double for each
  t.push(tick_count::now());
  for(decltype(v)::iterator k = v.begin(); k != v.end(); ++k) { *k = svrng_generate_double(engine, distr); }
  t.push(tick_count::now());

  int st = svrng_get_status();
  if(st != SVRNG_STATUS_OK) { printf("RNG FAILED: status error %i\n", st); goto lx; }

  // sum, mean, median, mad
  t.push(tick_count::now());
  for(decltype(v)::const_iterator k = v.begin(); k != v.end(); ++k) { v_sum += *k; }
  t.push(tick_count::now());

  v_mean = (v_sum / v.size());

  t.push(tick_count::now());
  parallel_sort(v);
  t.push(tick_count::now());

  v_median = v[(v.size() / 2L)];

  t.push(tick_count::now());
  parallel_for_each(v, [&](auto &elem) { elem = fabs((elem - v_median)); });
  t.push(tick_count::now());

  t.push(tick_count::now());
  parallel_sort(v);
  t.push(tick_count::now());

  v_mad = v[(v.size() / 2L)];

  // print stats
  printf( "A) trial size                         %zu double-precision\n", v.size()       );
  printf( "1) for generate                       %.6fs\n", f1()                          );
  printf( "2) for reduce                         %.6fs\n", f1()                          );
  printf( "3) parallel_sort                      %.6fs\n", f1()                          );
  printf( "4) parallel_for_each                  %.6fs\n", f1()                          );
  printf( "5) parallel_sort                      %.6fs\n", f1()                          );
  printf( "1) sum: sum(v)                        %.17e\n", v_sum                         );
  printf( "2) mean: sum/size(v)                  %.11e\n", v_mean                        );
  printf( "3) median: sort(v)[size(v)/2]         %.11e\n", v_median                      );
  printf( "4) mad: sort(v-median)[size(v)/2]     %.11e\n", v_mad                         );
  // implementation-dependent arithmetic types
  printf( "a) Machine epsilon (f):               %e\n",  FLT_EPSILON                     );
  printf( "b) Machine epsilon (ff):              %e\n",  DBL_EPSILON                     );
  printf( "c) Machine epsilon (fff):             %Le\n", LDBL_EPSILON                    );
  printf( "d) Machine rounds style:              %i\n",  FLT_ROUNDS                      );

lx:
  svrng_delete_distribution(distr);
  svrng_delete_engine(engine);
  return st;
}

int main() {
  parallel_invoke(fn, [](){});
  return 0;
}
