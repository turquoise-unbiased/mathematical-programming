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
  T t0, t1;
  Q q;
public:
  virtual void push(const T t) { q.push(t); }  // proxy with private
  virtual double f1() {
    return ((int(q.try_pop(t0)) & int(q.try_pop(t1))) ? (t1-t0).seconds() : -(0e0)); }  // time span
};

// RNG template class
template<size_t R, unsigned S = 37u>
class RNG {
  svrng_engine_t engine;
  svrng_distribution_t distr;
public:
  RNG() { engine = svrng_new_rand_engine(S);
          distr  = svrng_new_uniform_distribution_double(1e0, (R / 2e0)); }
  ~RNG() { svrng_delete_distribution(distr);
           svrng_delete_engine(engine); }
  double unif() const { return svrng_generate_double(engine, distr); }  // proxy with private
  svrng_double32_t unif32() const { return svrng_generate32_double(engine, distr); }  // proxy with private
};

// trial namespace
namespace trial {
  const size_t size = 1'000'000L;       // trial::size
  enum FUSION         { ON, OFF };      // loop fusion constants
  const int fusion  = FUSION::OFF;      // loop fusion constant
  int st            = SVRNG_STATUS_OK;  // RNG status
  int fn( void );
}

int trial::fn( void ) {
  // diagnostic
  bench bc;
  // random number generator
  RNG<trial::size> r;
  // container vector
  std::vector<double, scalable_allocator<double>> v((trial::size | 1L));
  double v_sum = 0e0;
  // double for each
  switch (fusion) {
    case FUSION::ON:
      bc.push(tick_count::now());  // 1&2)
      for(decltype(v)::iterator k = v.begin(); k < v.end(); ++k) {
        *k = r.unif();
        v_sum += *k;
      }
      bc.push(tick_count::now());

      st = svrng_get_status();
    break;
    default:
      const auto rem = (v.size() % 32L);
      const auto end = (rem ? (v.end() - 32L) : v.end());

      bc.push(tick_count::now());  // 1)
      for(decltype(v)::const_iterator k = v.begin(); k < end; k += 32L) {
        *((svrng_double32_t*)(&(*k))) = r.unif32();
      }
      // remainder
      for(decltype(v)::iterator k = (v.end() - rem); k < v.end(); ++k) {
        *k = r.unif();
      }
      bc.push(tick_count::now());

      st = svrng_get_status();
      if(st != SVRNG_STATUS_OK) { break; }

      // sum
      bc.push(tick_count::now());  // 2)
      for(decltype(v)::const_iterator k = v.begin(); k < v.end(); ++k) { v_sum += *k; }
      bc.push(tick_count::now());
  }

  // mean, median, mad
  // computing or jump according with RNG status
  switch (st) {
    case (not SVRNG_STATUS_OK):
      printf("RNG FAILED: status error %i\n", st);
    break;
    default:
      const auto v_mean = (v_sum / v.size());

      bc.push(tick_count::now());  // 3)
      parallel_sort(v);
      bc.push(tick_count::now());

      const auto v_median = v[(v.size() / 2L)];

      bc.push(tick_count::now());  // 4)
      parallel_for_each(v, [&](auto &elem) { elem = fabs((elem - v_median)); });
      bc.push(tick_count::now());

      bc.push(tick_count::now());  // 5)
      parallel_sort(v);
      bc.push(tick_count::now());

      const auto v_mad = v[(v.size() / 2L)];

      // print stats
      printf( "A) trial size                         %zu double-precision\n", v.size()       );
    if(fusion == FUSION::ON) {
      printf( "1&2) for generate [fused reduce]      %.6fs\n", bc.f1()                       );
    }else{
      printf( "1) for generate                       %.6fs\n", bc.f1()                       );
      printf( "2) for reduce                         %.6fs\n", bc.f1()                       );
    }
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
  }
  return st;
}

int main() {
  parallel_invoke(trial::fn, [](){});
  return 0;
}
