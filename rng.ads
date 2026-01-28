--  2020, Wojciech Lawren, All rights reserved.
--  SPARK interface for x86 RNG (Cryptographic Co-Processor).
--  [GNAT 14.1.0]
pragma Ada_2022;
pragma Profile (Ravenscar);

pragma Restrictions (
   No_Access_Parameter_Allocators,
   No_Coextensions,
   No_Recursion);

--  RDRAND RDSEED
package rng with
   No_Elaboration_Code_All,
   Pure,
   SPARK_Mode is

   --  generic type
   type rd64 is mod 2 ** 64 with Size => 64;

   --  rdrand
   generic
      type rx is mod <> or use rd64;
   function rand return rx with
      Pre     => (rx'Size in 64 | 32 | 16),  --  type check
      Post    => (rand'Result /= 0),  --  value check
      Global  => null;  --  global aspect

   --  rdseed
   generic
      type sx is mod <> or use rd64;
   function seed return sx with
      Pre     => (sx'Size in 64 | 32 | 16),  --  type check
      Post    => (seed'Result /= 0),  --  value check
      Global  => null;  --  global aspect

end rng;
