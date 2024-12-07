/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms c++17 c++20 & lambdas c++11 */
#include <oneapi/tbb.h>
#define MATHLIB_STANDALONE
#include <Rmath.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <execution>
#include <functional>
#include <numeric>
#include <span>
#include <vector>

constexpr auto N = 1'000'000L;
using namespace oneapi::tbb;

void fn() {
  // diagnostic
  static tick_count t0, t1;
  static concurrent_queue<tick_count> t;
  constexpr auto f1 = [&]() { { t.try_pop(t0); t.try_pop(t1); } return ((t1 - t0).count() / 1e+03); };
  // random number generator
  set_seed(37u, 59u);
  speculative_spin_mutex m;
  auto roller = [&](const double a = 1e0, const double b = (N / 2e0)) {
    speculative_spin_mutex::scoped_lock lock_shared(m); return runif(a, b); };
  // container vector
  std::vector<double> vd((N | 1L));
  const auto v = std::span<double>(vd);
  // parallel generate roller
  t.push(tick_count::now());
  std::generate(std::execution::par, v.begin(), v.end(), roller);
  t.push(tick_count::now());

  // sum, mean, median, mad
  t.push(tick_count::now());
  const auto v_sum = std::reduce(std::execution::par, v.begin(), v.end(), 0e0, std::plus<double>());
  t.push(tick_count::now());

  const auto v_mean = v_sum / v.size();

  t.push(tick_count::now());
  std::stable_sort(std::execution::par, v.begin(), v.end(), std::less<double>());
  t.push(tick_count::now());

  const auto v_median = v[(v.size() / 2L)];

  t.push(tick_count::now());
  std::transform(std::execution::par, v.begin(), v.end(), v.begin(),
                 [&](const auto &elem) { return fabs(elem - v_median); });
  t.push(tick_count::now());

  t.push(tick_count::now());
  std::stable_sort(std::execution::par, v.begin(), v.end(), std::less<double>());
  t.push(tick_count::now());

  const auto v_mad = v[(v.size() / 2L)];

  // print stats
  printf( "A) trial size (ff)     [el]:          %zu\n",  v.size()                       );
  printf( "1) parallel generate   [us]:          %.3f\n", f1()                           );
  printf( "2) parallel reduce     [us]:          %.3f\n", f1()                           );
  printf( "3) parallel stable_sort[us]:          %.3f\n", f1()                           );
  printf( "4) parallel transform  [us]:          %.3f\n", f1()                           );
  printf( "5) parallel stable_sort[us]:          %.3f\n", f1()                           );
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

int main() {
  parallel_invoke(fn, [](){});
  return 0;
}
