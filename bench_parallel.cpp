/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms c++17 & lambdas c++11 */
#include <stdio.h>
#include <float.h>
#include <fenv.h>
#include <mathimf.h>
#include <svrng.h>
#include <oneapi/tbb.h>
#include <oneapi/tbb/scalable_allocator.h>
#include <oneapi/tbb/flow_graph.h>

#define SVX (2L << 4L)  // short vector elements 2^n
typedef double svxdf_t __attribute__ ((vector_size ((SVX * sizeof(double)))));  // short vector arithmetic type

using namespace oneapi::tbb;

// tpl namespace
namespace tpl {
  // benchmark template class
  template<typename T = tick_count, typename Q = concurrent_queue<T>>
  class bench {
    T t0, t1; Q q;
  public:
    virtual void push(const T t) { q.push(t); }  // proxy with private
    virtual const double f1() {
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
    // proxy with private
    #if not (SVX % 32L)
      using svrngx_t = svrng_double32_t;
      svrngx_t unifsvx() const { return svrng_generate32_double(engine, distr); }
    #elif not (SVX % 16L)
      using svrngx_t = svrng_double16_t;
      svrngx_t unifsvx() const { return svrng_generate16_double(engine, distr); }
    #elif not (SVX % 8L)
      using svrngx_t = svrng_double8_t;
      svrngx_t unifsvx() const { return svrng_generate8_double(engine, distr); }
    #elif not (SVX % 4L)
      using svrngx_t = svrng_double4_t;
      svrngx_t unifsvx() const { return svrng_generate4_double(engine, distr); }
    #elif not (SVX % 2L)
      using svrngx_t = svrng_double2_t;
      svrngx_t unifsvx() const { return svrng_generate2_double(engine, distr); }
    #else
      using svrngx_t = double;
      svrngx_t unifsvx() const { return svrng_generate_double(engine, distr); }
    #endif
  };

  // median of vector template function
  template<typename T>
  const T median(const T* begin, const T* end, const size_t size) {
    switch (size) {
      case 0L: return -(0e0);
      case 1L: return *begin;
      default:
        const T* a = (end - ((size / 2L) + 1L));
        switch (size % 2L) {
          case 0L: return ((*a + *(a + 1L)) / 2e0);
          default: return *a;
  }}}

  // mean of vector template function
  template<typename T>
  const T mean(const T sum, const size_t size) {
    switch (size) {
      case 0L: return -(0e0);
      default: return (sum / size);
  }}

  // vector implementation template class
  template<typename T, size_t S>
  class vec {
    void* ptr;  // vector pointer
  public:
    const size_t size = S;  // vector size
    T* begin;   // vector pointer begin
    T* end;     // vector pointer end
    // stats
    T sum, mean, median, mad;

    vec() {
      ptr = scalable_calloc(size, sizeof(T));
      begin = (T*)ptr;
      end = (begin + size);
      sum = mean = median = mad = 0e0;
    }
    ~vec() { scalable_free(ptr); }

    template<typename Q>
    const size_t rem() const { return (size % (sizeof(Q) / sizeof(T))); }  // reminder

    template<typename Q>
    T* v_mod() const { return (end - rem<Q>()); }  // vector modulus

    // union modulus template function
    template<typename Q>
    const Q* sv_mod() const {
      const size_t rem = this->rem<Q>();
      const Q* end = (Q*)(&(*this->end));
      switch (rem) {
        case 0L: return end;
        default:
          switch ((sizeof(Q) ^ sizeof(T))) {
            case 0L: return (end - rem);
            default: return (end - 1L);
    }}}
  };
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
  using svrngx_t = decltype(rng)::svrngx_t;
  // container vector, pointers, reducers
  tpl::vec<double, trial::size> vec;
  // svrng pointers
  svrngx_t* rd_begin = (svrngx_t*)(&(*vec.begin));
  const svrngx_t* rd_end = vec.sv_mod<svrngx_t>();  // SVRNG modulus
  double* vec_rd = vec.v_mod<svrngx_t>();  // vector modulus
  // short vector pointers
  const svxdf_t* sv_begin = (svxdf_t*)(&(*vec.begin));
  const svxdf_t* sv_end = vec.sv_mod<svxdf_t>();  // short vector modulus
  const double* vec_sv = vec.v_mod<svxdf_t>();  // vector modulus
  // reducers
  svxdf_t sv_sum = { 0e0 };  // short vector reducer
  const double* svs_0 = (double*)(&sv_sum);  // short vector reducer pointers
  const double* svs_f = (svs_0 + (sizeof(svxdf_t) / sizeof(double)));
  // double for each
  switch (fusion) {
    case FUSION::ON:
      bench.push(tick_count::now());  // 1&2)
      for(double* k = vec.begin; k < vec.end; ++k) {
        *k = rng.unif();
        vec.sum += *k;
      }
      bench.push(tick_count::now());

      st = svrng_get_status();
    break;
    default:
      bench.push(tick_count::now());  // 1)
      for(svrngx_t* k = rd_begin; k < rd_end; ++k) { *k = rng.unifsvx(); }
      for(double* k = vec_rd; k < vec.end; ++k) { *k = rng.unif(); }  // remainder
      bench.push(tick_count::now());

      if(( st = svrng_get_status() ) != SVRNG_STATUS_OK) { break; }

      // sum
      bench.push(tick_count::now());  // 2)
      for(const svxdf_t* k = sv_begin; k < sv_end; ++k) { sv_sum += *k; }
      for(const double* k = svs_0; k < svs_f; ++k) { vec.sum += *k; }
      for(const double* k = vec_sv; k < vec.end; ++k) { vec.sum += *k; }  // remainder
      bench.push(tick_count::now());
  }

  // mean, median, mad
  // computing or jump according with RNG status
  switch (st) {
    case (not SVRNG_STATUS_OK):
      printf("RNG FAILED: status error %i\n", st);
    break;
    default:
      vec.mean = tpl::mean(vec.sum, vec.size);

      bench.push(tick_count::now());  // 3)
      parallel_sort(vec.begin, vec.end);
      bench.push(tick_count::now());

      vec.median = tpl::median(vec.begin, vec.end, vec.size);

      bench.push(tick_count::now());  // 4)
      parallel_for_each(vec.begin, vec.end, [&](auto &elem) { elem = fabs((elem - vec.median)); });
      bench.push(tick_count::now());

      bench.push(tick_count::now());  // 5)
      parallel_sort(vec.begin, vec.end);
      bench.push(tick_count::now());

      vec.mad = tpl::median(vec.begin, vec.end, vec.size);

      // print stats
      printf( "%zu) trial size:                        %zu double-precision\n", xch, vec.size);
    if(fusion == FUSION::ON) {
      printf( "1&2) for generate [fused reduce]      %.6fs\n", bench.f1()                    );
    }else{
      printf( "1) for generate                       %.6fs\n", bench.f1()                    );
      printf( "2) for reduce                         %.6fs\n", bench.f1()                    );
    }
      printf( "3) parallel_sort                      %.6fs\n", bench.f1()                    );
      printf( "4) parallel_for_each                  %.6fs\n", bench.f1()                    );
      printf( "5) parallel_sort                      %.6fs\n", bench.f1()                    );
      printf( "1) sum: sum(v)                        %.23e\n", vec.sum                       );
      printf( "2) mean: sum/size(v)                  %.17e\n", vec.mean                      );
      printf( "3) median: sort(v)[med]               %.17e\n", vec.median                    );
      printf( "4) mad: sort(v-median)[med]           %.17e\n", vec.mad                       );
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
  // graph parallelism
  flow::graph graph;
  flow::function_node<size_t, int, flow::queueing> fn(graph, info::default_concurrency(),
                                                      [](const size_t v) { return trial::fn(v); });

  for(size_t i = 1L; i < 2L; ++i) { fn.try_put(i); }
  graph.wait_for_all();

  return 0;
}
