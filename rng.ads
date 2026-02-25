-- 2020, Wojciech Lawren, All rights reserved.
-- Ada interface for x86 RNG Cryptographic Co-Processor.
-- [GNU Ada (GCC) version 14.1.0]
pragma Ada_2022;
pragma Profile (Ravenscar);

pragma Restrictions (
  No_Access_Parameter_Allocators,
  No_Coextensions,
  No_Recursion);

-- RDRAND RDSEED
package rng with
  No_Elaboration_Code_All,
  Pure,
  SPARK_Mode
is
  -- generic type
  type t_m64 is mod 2 ** 64 with Size => 64;
  subtype s_modular is t_m64;

  -- rdrand
  generic
    type t_mx is mod <> or use s_modular;
  function rand return t_mx with
    Pre => t_mx'Size in 64 | 32 | 16 and then t_mx'Modulus = 2 ** t_mx'Size,  -- type check
    Post => rand'Result /= 0,  -- value check
    Global => null;  -- global aspect

  -- rdseed
  generic
    type t_mx is mod <> or use s_modular;
  function seed return t_mx with
    Pre => t_mx'Size in 64 | 32 | 16 and then t_mx'Modulus = 2 ** t_mx'Size,  -- type check
    Post => seed'Result /= 0,  -- value check
    Global => null;  -- global aspect

end rng;
