/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms and short vectors using double precision floating-point values
   [icpx 2025.0.1]
*/
#include <svrng.h>

#include <oneapi/tbb/scalable_allocator.h>
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/tick_count.h>

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
    const svrng_engine_t engine;
    const svrng_distribution_t distr;
  public:
    RNG(const double r)
      : engine( svrng_new_rand_engine(S) )
      , distr( svrng_new_uniform_distribution_double(0e0, r) ) {}
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
    #endif
  };

  // vector implementation template class
  template<typename T>
  class vector {
    void* const ptr;  // continuous virtual memory' address
  public:
    const size_t size;  // vector size
    T* const begin;   // vector pointer begin
    T* const end;     // vector pointer end
    const int st;  // vec status

    vector(const size_t r)
      : ptr( scalable_calloc(r, sizeof(T)) )
      , size( r )
      , begin( (T*)ptr )
      , end( (begin + size) )
      , st( (ptr ? TBBMALLOC_OK : TBBMALLOC_NO_EFFECT) ) {}
    ~vector() { scalable_free(ptr); }

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

  // median of sorted vector template function
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

  // mean of vector (sum<T>/size) template function
  template<typename T>
  const T mean(const T sum, const size_t size) {
    switch (size) {
      case 0L: return -(0e0);
      default: return (sum / size);
  }}
}  // end tpl
