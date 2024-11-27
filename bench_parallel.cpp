/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms c++17 & lambdas c++11 */
#include <oneapi/tbb.h>

#include <algorithm>
#include <array>
#include <execution>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

#define N 100000L
using namespace oneapi::tbb;

int main() {
  // diagnostic
  std::array<tick_count, 10L> t;
  auto f1 = [=](const tick_count t0, const tick_count t1) { return  ((t1 - t0).count() / 1e+03); };
  auto println = [=](const auto rem, const auto score) { std::cout << rem << score << std::endl; };
  // random number generator
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(1L, (N >> 1L));
  std::function<double()> roller = [&]() { return distribution(generator); };
  // container vector
  std::vector<double> v(((N & 1L) ? N : (N | 1L)));
  // parallel transform roller
  t[0L] = tick_count::now();
  std::transform(std::execution::par, v.cbegin(), v.cend(), v.begin(),
                 [&](const auto &elem) { return roller(); });
  t[1L] = tick_count::now();

  // sum, mean, median, mad
  t[2L] = tick_count::now();
  const auto v_sum = std::reduce(std::execution::par, v.cbegin(), v.cend());
  t[3L] = tick_count::now();

  const auto v_mean = v_sum / static_cast<double>(v.size());

  t[4L] = tick_count::now();
  std::sort(std::execution::par, v.begin(), v.end());
  t[5L] = tick_count::now();

  const auto v_median = v.at((v.size() / 2L));

  t[6L] = tick_count::now();
  std::transform(std::execution::par, v.cbegin(), v.cend(), v.begin(),
                 [=](const auto &elem) { return std::abs(elem - v_median); });
  t[7L] = tick_count::now();

  t[8L] = tick_count::now();
  std::sort(std::execution::par, v.begin(), v.end());
  t[9L] = tick_count::now();

  const auto v_mad = v.at((v.size() / 2L));

  // print stats
  println( "A) trial size (double) [el]:        ", v.size()                );
  println( "1) parallel transform  [us]:        ", f1(t[0L], t[1L])        );
  println( "2) parallel reduce     [us]:        ", f1(t[2L], t[3L])        );
  println( "3) parallel sort       [us]:        ", f1(t[4L], t[5L])        );
  println( "4) parallel transform  [us]:        ", f1(t[6L], t[7L])        );
  println( "5) parallel sort       [us]:        ", f1(t[8L], t[9L])        );
  println( "1) sum: sum(v)                      ", v_sum                   );
  println( "2) mean: sum/size(v)                ", v_mean                  );
  println( "3) median: sort(v)[size(v)/2]       ", v_median                );
  println( "4) mad: sort(v-median)[size(v)/2]   ", v_mad                   );
  // 0 = OK
  return 0L;
}
