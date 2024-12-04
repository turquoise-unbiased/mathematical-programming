/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms c++17 c++20 & lambdas c++11 */
#include <oneapi/tbb.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <execution>
#include <functional>
#include <numeric>
#include <random>
#include <span>
#include <vector>

constexpr auto N = 100000L;
using namespace oneapi::tbb;

void fn() {
  // diagnostic
  static tick_count t0, t1;
  static concurrent_queue<tick_count> t;
  constexpr auto f1 = [&]() { { t.try_pop(t0); t.try_pop(t1); } return ((t1 - t0).count() / 1e+03); };
  // random number generator
  speculative_spin_mutex m;
  std::default_random_engine generator(37u);
  std::uniform_real_distribution<double> distribution(1e0, (N / 2e0));
  std::function<double()> roller = [&]() {
    speculative_spin_mutex::scoped_lock lock_shared(m); return distribution(generator); };
  // container vector
  std::vector<double> vd((N | 1L));
  const auto v = std::span<double>(vd);
  // parallel generate roller
  t.push(tick_count::now());
  std::generate(std::execution::par, v.begin(), v.end(), roller);
  t.push(tick_count::now());

  // sum, mean, median, mad
  t.push(tick_count::now());
  const auto v_sum = std::reduce(std::execution::par, v.begin(), v.end());
  t.push(tick_count::now());

  const auto v_mean = v_sum / v.size();

  t.push(tick_count::now());
  parallel_sort(v);
  t.push(tick_count::now());

  const auto v_median = v[(v.size() / 2L)];

  t.push(tick_count::now());
  parallel_for_each(v, [=](auto &elem) { elem = abs(elem - v_median); });
  t.push(tick_count::now());

  t.push(tick_count::now());
  parallel_sort(v);
  t.push(tick_count::now());

  const auto v_mad = v[(v.size() / 2L)];

  // print stats
  printf( "A) trial size (ff)     [el]:          %zu\n",  v.size()                       );
  printf( "1) parallel_for_each   [us]:          %.3f\n", f1()                           );
  printf( "2) parallel_reduce     [us]:          %.3f\n", f1()                           );
  printf( "3) parallel_sort       [us]:          %.3f\n", f1()                           );
  printf( "4) parallel_for_each   [us]:          %.3f\n", f1()                           );
  printf( "5) parallel_sort       [us]:          %.3f\n", f1()                           );
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
  parallel_invoke(fn, [](){});
  return 0;
}
