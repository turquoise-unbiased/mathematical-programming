-- 2020, Wojciech Lawren, All rights reserved.
-- Ada interface for x86 RNG Cryptographic Co-Processor.
-- [GNU Ada (GCC) version 14.1.0]
pragma Ada_2022;

with System.Machine_Code;  -- Assembler

-- RDRAND RDSEED
package body rng with
  SPARK_Mode
is
  -- retry limit
  subtype s_lim is Positive range 1 .. 10;
  RL : constant s_lim := 2;
  use ASCII;  -- LF HT

  function rand return t_mx with SPARK_Mode => Off is
  begin
    return r : t_mx do  -- return value
      System.Machine_Code.Asm  -- x86 att
        (Template => "xorl %%eax, %%eax"    & LF & HT &
                     "movl %1, %%ecx"       & LF & HT &
                     "1:"                   & LF & HT &
                     "rdrand %0"            & LF & HT &
                     "jc 2f"                & LF & HT &
                     "loop 1b"              & LF & HT &
                     "cmovncl %%ecx, %%eax" & LF & HT &
                     "2:",
         Outputs  => t_mx'Asm_Output ("=a", r),
         Inputs   => s_lim'Asm_Input ("n", RL),
         Clobber  => "rcx, cc",
         Volatile => True);
    end return;
  end rand;

  function seed return t_mx with SPARK_Mode => Off is
  begin
    return r : t_mx do  -- return value
      System.Machine_Code.Asm  -- x86 att
        (Template => "xorl %%eax, %%eax"    & LF & HT &
                     "movl %1, %%ecx"       & LF & HT &
                     "1:"                   & LF & HT &
                     "rdseed %0"            & LF & HT &
                     "jc 2f"                & LF & HT &
                     "loop 1b"              & LF & HT &
                     "cmovncl %%ecx, %%eax" & LF & HT &
                     "2:",
         Outputs  => t_mx'Asm_Output ("=a", r),
         Inputs   => s_lim'Asm_Input ("n", RL),
         Clobber  => "rcx, cc",
         Volatile => True);
    end return;
  end seed;

end rng;
