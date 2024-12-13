/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms c++17 & lambdas c++11 */
#include <float.h>
#include <mathimf.h>
#include <svrng.h>
#include <oneapi/tbb.h>
#include <oneapi/tbb/scalable_allocator.h>

#include <cstdio>
#include <vector>

using namespace oneapi::tbb;

// benchmark template class
template<typename T = tick_count, typename Q = concurrent_queue<T>>
class bench {
  T t0, t1; Q q;
public:
  void push(const T t) { this->q.push(t); }  // proxy with private
  double f1() { return ((int(this->q.try_pop(this->t0)) & int(this->q.try_pop(this->t1))) ?
    (this->t1-this->t0).seconds() : -(0e0)); }  // time span
};

// trial namespace
namespace trial {
  const long size = 1'000'000L;
  int fn( void );
}

int trial::fn( void ) {
  // diagnostic
  bench bc;
  // random number generator
  svrng_engine_t engine      = svrng_new_rand_engine(37u);
  svrng_distribution_t distr = svrng_new_uniform_distribution_double(1e0, (trial::size / 2e0));
  // container vector
  std::vector<double, scalable_allocator<double>> v((trial::size | 1L));
  double v_sum = 0e0, v_mean = 0e0, v_median = 0e0, v_mad = 0e0;
  // double for each
  bc.push(tick_count::now());  // 1)
  for(decltype(v)::iterator k = v.begin(); k != v.end(); ++k) { *k = svrng_generate_double(engine, distr); }
  bc.push(tick_count::now());

  int st = svrng_get_status();  // computing or jump according with RNG status
  if(st != SVRNG_STATUS_OK) { printf("RNG FAILED: status error %i\n", st); goto lx; }

  // sum, mean, median, mad
  bc.push(tick_count::now());  // 2)
  for(decltype(v)::const_iterator k = v.begin(); k != v.end(); ++k) { v_sum += *k; }
  bc.push(tick_count::now());

  v_mean = (v_sum / v.size());

  bc.push(tick_count::now());  // 3)
  parallel_sort(v);
  bc.push(tick_count::now());

  v_median = v[(v.size() / 2L)];

  bc.push(tick_count::now());  // 4)
  parallel_for_each(v, [&](auto &elem) { elem = fabs((elem - v_median)); });
  bc.push(tick_count::now());

  bc.push(tick_count::now());  // 5)
  parallel_sort(v);
  bc.push(tick_count::now());

  v_mad = v[(v.size() / 2L)];

  // print stats
  printf( "A) trial size                         %zu double-precision\n", v.size()       );
  printf( "1) for generate                       %.6fs\n", bc.f1()                       );
  printf( "2) for reduce                         %.6fs\n", bc.f1()                       );
  printf( "3) parallel_sort                      %.6fs\n", bc.f1()                       );
  printf( "4) parallel_for_each                  %.6fs\n", bc.f1()                       );
  printf( "5) parallel_sort                      %.6fs\n", bc.f1()                       );
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
  parallel_invoke(trial::fn, [](){});
  return 0;
}
