/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms c++17 & lambdas c++11 */
#include <oneapi/tbb.h>
#include <oneapi/mkl/rng.hpp>
#include <sycl/sycl.hpp>

#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

typedef double F;  // float, double, long double
constexpr auto N = 100000L;
constexpr auto SEED = 37u;
using namespace oneapi::tbb;
using namespace oneapi::mkl;

void fn() {
  // container vector
  sycl::queue queue;  // {sycl::default_selector_v};
  std::vector<F> vd((N | 1L));
  const auto v = sycl::span<F>(vd);
  typedef decltype(v)::iterator R;
  // random number generator
  rng::default_engine engine(queue, SEED);
  rng::uniform<F> distr(1e0, (N / 2e0));
  // diagnostic
  static std::array<tick_count, 10L> t;
  constexpr auto f1 = [=](const tick_count t0, const tick_count t1) { return ((t1 - t0).count() / 1e+03); };
  // parallel generate
  t[0L] = tick_count::now();
  {
    sycl::buffer<F, 1> r_buf(v.data(), v.size());
    rng::generate(distr, engine, v.size(), r_buf);
  }
  t[1L] = tick_count::now();

  // sum, mean, median, mad
  t[2L] = tick_count::now();
  const auto v_sum = parallel_reduce(
    blocked_range<R>(v.begin(), v.end()),
    0e0,
    [&](blocked_range<R> &r, F partial_sum) {
      return std::accumulate(r.begin(), r.end(), partial_sum);
    },
    std::plus<F>());
  t[3L] = tick_count::now();

  const auto v_mean = v_sum / v.size();

  t[4L] = tick_count::now();
  parallel_sort(v);
  t[5L] = tick_count::now();

  const auto v_median = v[(v.size() / 2L)];

  t[6L] = tick_count::now();
  parallel_for_each(v, [=](auto &elem) { elem = abs(elem - v_median); });
  t[7L] = tick_count::now();

  t[8L] = tick_count::now();
  parallel_sort(v);
  t[9L] = tick_count::now();

  const auto v_mad = v[(v.size() / 2L)];

  // print stats
  printf( "A) trial size (ff)     [el]:          %zu\n",  v.size()                       );
  printf( "1) parallel generate   [us]:          %.3f\n", f1(t[0L], t[1L])               );
  printf( "2) parallel_reduce     [us]:          %.3f\n", f1(t[2L], t[3L])               );
  printf( "3) parallel_sort       [us]:          %.3f\n", f1(t[4L], t[5L])               );
  printf( "4) parallel_for_each   [us]:          %.3f\n", f1(t[6L], t[7L])               );
  printf( "5) parallel_sort       [us]:          %.3f\n", f1(t[8L], t[9L])               );
  printf( "1) sum: sum(v)                        %.3f\n", v_sum                          );
  printf( "2) mean: sum/size(v)                  %.3f\n", v_mean                         );
  printf( "3) median: sort(v)[size(v)/2]         %.3f\n", v_median                       );
  printf( "4) mad: sort(v-median)[size(v)/2]     %.3f\n", v_mad                          );
  // implementation-dependent arithmetic types
  printf( "a) Machine epsilon (f):               %e\n",  FLT_EPSILON                     );
  printf( "b) Machine epsilon (ff):              %e\n",  DBL_EPSILON                     );
  printf( "c) Machine epsilon (fff):             %Le\n", LDBL_EPSILON                    );
  printf( "d) Machine rounds style:              %i\n",  FLT_ROUNDS                      );
}

int main() {
  parallel_invoke(fn, [](){});  // evaluates functions in parallel in bound context
  return 0;
}
