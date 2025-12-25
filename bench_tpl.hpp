/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms and short vectors using double precision floating-point values
   [icpx 2025.0.1]
*/
#include <math.h>
#include "SFMT.h"  // https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/SFMT/index.html [1.5.1]

// tpl namespace
namespace tpl {
  constexpr size_t SVX = (2L << 8L);  // short vector elements 2^n [>= (SFMT_MEXP / 128 + 1) * 2]
  // short vector RNG types
  typedef double svxdf_t __attribute__ ((vector_size ((SVX * sizeof(double)))));
  typedef unsigned long svxdu_t __attribute__ ((vector_size ((SVX * sizeof(unsigned long)))));
  static_assert((sizeof(svxdf_t) == sizeof(svxdu_t)));  // for union

  // RNG template class
  template<unsigned S = 10u>  // <= init_key.size + 1
  class RNG {
    sfmt_t* const sfmt = new sfmt_t;  // SFMT internal state
    const double R;
    const unsigned init_key[9u] = {37u, 41u, 43u, 47u, 53u, 59u, 61u, 67u, 71u};
  public:
    RNG(const int k, const double r)
      : R( scalbln(r, -(53L)) )
      { sfmt_init_by_array(sfmt, (unsigned*)(&init_key), (k%S)); }
    ~RNG() { delete sfmt; }
    double unif() const { return ((sfmt_genrand_uint64(sfmt) >> 11L) * R); }  // 53-bit resolution * r
    using svrngx_t = svxdf_t;
    void unifsvx(svrngx_t* const vf) const {
      sfmt_fill_array64(sfmt, (unsigned long*)vf, SVX);
      *vf = (__builtin_convertvector((*((svxdu_t*)vf) >> 11L), svxdf_t) * R); }  // 53-bit resolution * r
  };

  // vector implementation template class
  template<typename T>
  class vector {
    T* const ptr;  // continuous virtual memory' address
  public:
    const size_t size;  // vector size
    T* const begin;   // vector pointer begin
    T* const end;     // vector pointer end

    vector(const size_t r)
      : ptr( new T[r] { 0e0 } )
      , size( r )
      , begin( (T*)ptr )
      , end( (begin + size) ) {}
    ~vector() { delete [] ptr; }

    template<typename Q>
    const size_t rem() const { return (size % (sizeof(Q) / sizeof(T))); }  // reminder

    template<typename Q>
    T* v_mod() const { return (end - rem<Q>()); }  // vector modulus T*

    // union modulus Q*
    template<typename Q>
    const Q* sv_mod() const {
      const size_t rem = this->rem<Q>();
      const Q* const end = (Q*)(&(*this->end));
      return ( ((not rem) or (sizeof(Q)==sizeof(T))) ? (end-rem) : (end-1L) ); }
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
