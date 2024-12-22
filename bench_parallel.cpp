/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms c++17 & lambdas c++11 */
#include <float.h>
#include <fenv.h>
#include <mathimf.h>
#include <svrng.h>
#include <oneapi/tbb.h>
#include <oneapi/tbb/scalable_allocator.h>
#include <oneapi/tbb/flow_graph.h>

#include <cstdio>
#include <vector>

typedef double v32df __attribute__ ((vector_size (256)));

using namespace oneapi::tbb;

// tpl namespace
namespace tpl {
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

  // median of vector template function
  template<typename T>
  double median(const T r) {
    switch (r.size()) {
      case 0L: return -(0e0);
      case 1L: return *(r.begin());
      default:
        const auto a = (r.end() - ((r.size() / 2L) + 1L));
        switch (r.size() % 2L) {
          case 0L: return ((*a + *(a + 1L)) / 2e0);
          default: return *a;
  }}}

  // mean of vector template function
  template<typename T>
  double mean(const T r, const double sum) {
    switch (r.size()) {
      case 0L: return -(0e0);
      default: return (sum / r.size());
  }}
}  // end tpl

// trial namespace
namespace trial {
  const size_t size = 1'000'000L;       // trial::size
  enum FUSION         { ON, OFF };      // loop fusion constants
  const int fusion  = FUSION::OFF;      // loop fusion constant
  volatile int st   = SVRNG_STATUS_OK;  // RNG status
  int fn(const size_t xch);
}  // end trial

int trial::fn(const size_t xch) {
  // diagnostic
  tpl::bench bench;
  // random number generator
  tpl::RNG<trial::size> rng;
  // container vector
  std::vector<double, scalable_allocator<double>> vec(trial::size);
  svrng_double32_t* sv_begin = (svrng_double32_t*)(&(*vec.begin()));
  svrng_double32_t* sv_end   = (svrng_double32_t*)(&(*vec.end()));
  v32df* df_begin = (v32df*)(&(*vec.begin()));
  v32df* df_end   = (v32df*)(&(*vec.end()));
  v32df df_sum = { 0e0 };
  double v_sum = 0e0;
  // double for each
  switch (fusion) {
    case FUSION::ON:
      bench.push(tick_count::now());  // 1&2)
      for(decltype(vec)::iterator k = vec.begin(); k < vec.end(); ++k) {
        *k = rng.unif();
        v_sum += *k;
      }
      bench.push(tick_count::now());

      st = svrng_get_status();
    break;
    default:
      const auto rem_32 = (vec.size() % 32L);
      const auto end_sv = (rem_32 ? (sv_end - 1L) : sv_end);

      bench.push(tick_count::now());  // 1)
      for(svrng_double32_t* k = sv_begin; k < end_sv; ++k) { *k = rng.unif32(); }
      // remainder
      for(decltype(vec)::iterator k = (vec.end() - rem_32); k < vec.end(); ++k) { *k = rng.unif(); }
      bench.push(tick_count::now());

      if(( st = svrng_get_status() ) != SVRNG_STATUS_OK) { break; }

      // sum
      const auto end_df = (rem_32 ? (df_end - 1L) : df_end);

      bench.push(tick_count::now());  // 2)
      for(v32df* k = df_begin; k < end_df; ++k) { df_sum += *k; }
      for(size_t i = 0L; i < 32L; ++i) { v_sum += df_sum[i]; }
      // remainder
      for(decltype(vec)::iterator k = (vec.end() - rem_32); k < vec.end(); ++k) { v_sum += *k; }
      bench.push(tick_count::now());
  }

  // mean, median, mad
  // computing or jump according with RNG status
  switch (st) {
    case (not SVRNG_STATUS_OK):
      printf("RNG FAILED: status error %i\n", st);
    break;
    default:
      const auto v_mean = tpl::mean(vec, v_sum);

      bench.push(tick_count::now());  // 3)
      parallel_sort(vec);
      bench.push(tick_count::now());

      const auto v_median = tpl::median(vec);

      bench.push(tick_count::now());  // 4)
      parallel_for_each(vec, [&](auto &elem) { elem = fabs((elem - v_median)); });
      bench.push(tick_count::now());

      bench.push(tick_count::now());  // 5)
      parallel_sort(vec);
      bench.push(tick_count::now());

      const auto v_mad = tpl::median(vec);

      // print stats
      printf( "%zu) trial size:                        %zu double-precision\n", xch, vec.size());
    if(fusion == FUSION::ON) {
      printf( "1&2) for generate [fused reduce]      %.6fs\n", bench.f1()                    );
    }else{
      printf( "1) for generate                       %.6fs\n", bench.f1()                    );
      printf( "2) for reduce                         %.6fs\n", bench.f1()                    );
    }
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
      printf( "e) x87 FPU exception flags:           %i\n",  fetestexcept(FE_ALL_EXCEPT)     );
  }
  return st;
}

int main() {
  flow::graph graph;
  flow::function_node<size_t, int, flow::queueing> fn(graph, info::default_concurrency(),
                                                      [](const size_t v) { return trial::fn(v); });

  for(size_t i = 1L; i < 2L; ++i) { fn.try_put(i); }
  graph.wait_for_all();

  return 0;
}
