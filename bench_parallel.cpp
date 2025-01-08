/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms and short vectors using double precision floating-point values
*/
#include <stdio.h>
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <svrng.h>

#include <oneapi/tbb/scalable_allocator.h>
#include <oneapi/tbb/flow_graph.h>
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/tick_count.h>
#include <oneapi/tbb/parallel_for_each.h>
#include <oneapi/tbb/parallel_sort.h>

constexpr size_t SVX = (2L << 4L);  // short vector elements 2^n [>= 2]

typedef double svxdf_t __attribute__ ((vector_size ((SVX * sizeof(double)))));  // short vector arithmetic type

constexpr double MILLE = 1e6;  // trial_scale [>= SVX]

using namespace oneapi::tbb;

// tpl namespace
namespace tpl {
  // benchmark template class
  template<typename T = tick_count, typename Q = concurrent_queue<T>>
  class bench {
    Q q;
  public:
    virtual void push(const T t) { q.push(t); }  // proxy with private
    virtual const double f1() { T t0, t1;
      return ((int(q.try_pop(t0)) & int(q.try_pop(t1))) ? (t1-t0).seconds() : -(0e0)); }  // time span
  };

  // RNG template class
  template<unsigned S = 37u>
  class RNG {
    svrng_engine_t engine;
    svrng_distribution_t distr;
  public:
    int st = SVRNG_STATUS_OK;  // RNG status
    RNG(const double r) {
      engine = svrng_new_rand_engine(S);
      distr  = svrng_new_uniform_distribution_double(0e0, r);
      st = svrng_get_status();
    }
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
  const T median(const T* const begin, const T* const end, const size_t size) {
    switch (size) {
      case 0L: return -(0e0);
      case 1L: return *begin;
      default:
        const T* const a = (end - ((size / 2L) + 1L));
        switch ((size % 2L)) {
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
  template<typename T>
  class vec {
    void* const ptr;  // vector pointer
  public:
    const size_t size;  // vector size
    T* const begin;   // vector pointer begin
    T* const end;     // vector pointer end
    T sum, mean, median, mad;  // stats
    int st = TBBMALLOC_OK;  // vec status

    vec(const size_t r)
      : ptr( scalable_calloc(r, sizeof(T)) )
      , size( r )
      , begin( (T*)ptr )
      , end( (begin + size) )
    { if(not ptr) { st = TBBMALLOC_NO_EFFECT; } }
    ~vec() { scalable_free(ptr); }

    template<typename Q>
    const size_t rem() const { return (size % (sizeof(Q) / sizeof(T))); }  // reminder

    template<typename Q>
    T* v_mod() const { return (end - rem<Q>()); }  // vector modulus

    // union modulus template function
    template<typename Q>
    const Q* sv_mod() const {
      const size_t rem = this->rem<Q>();
      const Q* const end = (Q*)(&(*this->end));
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
  const int STATUS_OK = (SVRNG_STATUS_OK | TBBMALLOC_OK);  // constructors ok
  enum class FUSION { ON = 1, OFF };  // loop fusion constants
  size_t fn(const size_t N);
}  // end trial

size_t trial::fn(const size_t N) {
  FUSION fusion = FUSION::OFF;  // loop fusion constant
  double trial_scale = scalbln(MILLE, N);  // trial_scale
  size_t trial_size  = lrint(trial_scale);  // trial_size
  // diagnostic
  tpl::bench bench;
  // random number generator
  tpl::RNG rng(trial_scale);
  using svrngx_t = decltype(rng)::svrngx_t;
  // container vector, pointers, reducers
  tpl::vec<double> vec(trial_size);
  // double for each
  switch ( rng.st = svrng_get_status(); ( rng.st | vec.st ) ) {
    case STATUS_OK:
      // svrng pointers
      svrngx_t* const rd_begin = (svrngx_t*)(&(*vec.begin));
      const svrngx_t* const rd_end = vec.sv_mod<svrngx_t>();  // SVRNG modulus
      double* const vec_rd = vec.v_mod<svrngx_t>();  // vector modulus
      // short vector pointers
      const svxdf_t* const sv_begin = (svxdf_t*)(&(*vec.begin));
      const svxdf_t* const sv_end = vec.sv_mod<svxdf_t>();  // short vector modulus
      const double* const vec_sv = vec.v_mod<svxdf_t>();  // vector modulus
      // reducers
      svxdf_t sv_sum = { 0e0 };  // short vector reducer
      const double* const svs_0 = (double*)(&sv_sum);  // short vector reducer pointers
      const double* const svs_f = (svs_0 + (sizeof(svxdf_t) / sizeof(double)));
      // FUSION
      switch (fusion) {
        case FUSION::ON:
          bench.push(tick_count::now());  // 1&2)
          for(double* k = vec.begin; k < vec.end; ++k) {
            *k = rng.unif();
            vec.sum += *k;
          }
          bench.push(tick_count::now());
        break;
        case FUSION::OFF:
          bench.push(tick_count::now());  // 1)
          for(svrngx_t* k = rd_begin; k < rd_end; ++k) { *k = rng.unifsvx(); }
          for(double* k = vec_rd; k < vec.end; ++k) { *k = rng.unif(); }  // remainder
          bench.push(tick_count::now());

          if(( rng.st = svrng_get_status() ) != SVRNG_STATUS_OK) { break; }

          // sum
          bench.push(tick_count::now());  // 2)
          for(const svxdf_t* k = sv_begin; k < sv_end; ++k) { sv_sum += *k; }
          for(const double* k = svs_0; k < svs_f; ++k) { vec.sum += *k; }
          for(const double* k = vec_sv; k < vec.end; ++k) { vec.sum += *k; }  // remainder
          bench.push(tick_count::now());
        break;
      }  // FUSION
    break;
  }  // STATUS
  // mean, median, mad
  // computing or jump according with RNG status
  switch ( rng.st = svrng_get_status(); ( rng.st | vec.st ) ) {
    case STATUS_OK:
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
      printf( "1) sum: sum(v)                        %.23e\n", vec.sum                       );
      printf( "2) mean: sum/size(v)                  %.17e\n", vec.mean                      );
      printf( "3) median: sort(v)[med]               %.17e\n", vec.median                    );
      printf( "4) mad: sort(v-median)[med]           %.17e\n", vec.mad                       );
      // implementation-dependent arithmetic types
      printf( "a) Machine epsilon (f):               %e\n",  FLT_EPSILON                     );
      printf( "b) Machine epsilon (ff):              %e\n",  DBL_EPSILON                     );
      printf( "c) Machine epsilon (fff):             %Le\n", LDBL_EPSILON                    );
      printf( "d) Machine rounds style:              %i\n",  FLT_ROUNDS                      );
      printf( "e) x87 FPU exception flags:           %i\n\n", fetestexcept(FE_ALL_EXCEPT)    );
    break;
    default:
      printf( "APP FAILED: status rng: %i | tbb: %i\n", rng.st, vec.st );
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
    [&](const size_t &n) { return trial::fn(n); });
  // function_node
  flow::function_node<size_t, size_t, flow::queueing_lightweight> comp(graph, flow::serial,
    [&](const size_t &z) { return __atomic_add_fetch(&test_size, z, __ATOMIC_SEQ_CST); });
  // edge
  flow::make_edge(fn, comp);
  // execute incoming message
  for(size_t n = 0L; n < 1L; ++n) { fn.try_put(n); }
  // block & catch & throw
  try { graph.wait_for_all(); }
  catch (...) { graph.cancel(); throw; }
  // print info
  printf( "a) graph exception thrown:            %i\n", graph.exception_thrown()   );
  printf( "b) compound test size:                %zu\n", test_size                 );

  return 0;
}
