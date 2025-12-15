/* 2024, Wojciech Lawren, All rights reserved.
   benchmark parallel algorithms and short vectors using double precision floating-point values
   [icpx 2025.0.1]
*/
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/tick_count.h>

using namespace oneapi::tbb;

// tpl namespace
namespace tpl {
  // benchmark template class
  template<typename T = tick_count, typename Q = concurrent_queue<T>>
  class bench { Q q;  // concurrent_queue
  public:
    virtual void push(const T t) { q.push(t); }  // proxy with private
    virtual const double f1() { T t0, t1;
      return ((int(q.try_pop(t0)) & int(q.try_pop(t1))) ? (t1-t0).seconds() : -(0e0)); }  // time span
  };
}  // end tpl
