/* 2025, Wojciech Lawren, All rights reserved.
   c++ armadillo cube type simulation prototype
   [icpx 2025.0.1] [g++ 11.3.0]
*/
#include <armadillo>  // https://arma.sourceforge.net/download.html [15.2.3]

using namespace arma;

namespace trial {
  constexpr uword N = 1e3;  // vectors size
  constexpr uword Q = 9L;  // quantile size
}  // end trial

int main() {
  wall_clock timer;  // timer

  timer.tic();
  const cube C (5L, trial::N, 3L, fill::randn);  // vectors [horizontal] Gaussian distribution [split]
  timer.freeze();

  const vec qnt = linspace(0e0, 1e0, trial::Q);  // quantile [equidistant]
  cube R (C.n_rows, qnt.n_elem, C.n_slices);  // result cube

  uword idx = 0L; timer.unfreeze();
  C.each_slice( [&](const mat &K) { R.slice(idx++) = quantile(K, qnt, 1L); } );  // dim = 1
  const double f1 = timer.toc();  // seconds since tic

  R.print("cube C quantile");
  cout << "\n" << "elapsed: " << f1 << "s" << endl;

  return 0;
}
