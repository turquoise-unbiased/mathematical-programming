/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms c++17 & lambdas c++11 */
#include <oneapi/tbb.h>

#include <execution>
#include <iostream>

#define N 100000

using namespace std::execution;
using namespace oneapi::tbb;

int main() {
  // diagnostic
  std::array<tick_count, 10> t;
  auto f1 = [=](const tick_count t0, const tick_count t1) { return  ((t1 - t0).count() / 1e+03); };
  auto println = [=](const auto rem, const auto score) { std::cout << rem << score << std::endl; };
  // random number generator
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(1, (N >> 1));
  std::function<double()> roller = [&]() { return distribution(generator); };
  // container vector
  std::vector<double> v(((N & 1) ? N : (N | 1)));
  // parallel transform roller
  t[0] = tick_count::now();
  std::transform(par, v.cbegin(), v.cend(), v.begin(),
                 [&](const auto &elem) { return roller(); });
  t[1] = tick_count::now();

  // sum, mean, median, mad
  t[2] = tick_count::now();
  const auto v_sum = std::reduce(par, v.cbegin(), v.cend());
  t[3] = tick_count::now();

  const auto v_mean = v_sum / static_cast<double>(v.size());

  t[4] = tick_count::now();
  std::sort(par, v.begin(), v.end());
  t[5] = tick_count::now();

  const auto v_median = v.at((v.size() / 2));

  t[6] = tick_count::now();
  std::transform(par, v.cbegin(), v.cend(), v.begin(),
                 [=](const auto &elem) { return std::abs(elem - v_median); });
  t[7] = tick_count::now();

  t[8] = tick_count::now();
  std::sort(par, v.begin(), v.end());
  t[9] = tick_count::now();

  const auto v_mad = v.at((v.size() / 2));

  // print stats
  println( "A) trial size (double) [el]:        ", v.size()                );
  println( "1) parallel transform  [us]:        ", f1(t[0], t[1])          );
  println( "2) parallel reduce     [us]:        ", f1(t[2], t[3])          );
  println( "3) parallel sort       [us]:        ", f1(t[4], t[5])          );
  println( "4) parallel transform  [us]:        ", f1(t[6], t[7])          );
  println( "5) parallel sort       [us]:        ", f1(t[8], t[9])          );
  println( "1) sum: sum(v)                      ", v_sum                   );
  println( "2) mean: sum/size(v)                ", v_mean                  );
  println( "3) median: sort(v)[size(v)/2]       ", v_median                );
  println( "4) mad: sort(v-median)[size(v)/2]   ", v_mad                   );
  // 0 = OK
  return 0;
}
