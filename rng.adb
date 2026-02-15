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
  subtype rlim is Positive range 1 .. 10;
  RL : constant rlim := 2;
  use ASCII;  -- LF HT

  function rand return rx with SPARK_Mode => Off is
    r : rx;  -- return value
  begin
    System.Machine_Code.Asm  -- x86 att
      (Template => "xorl %%eax, %%eax"    & LF & HT &
                   "movl %1, %%ecx"       & LF & HT &
                   "1:"                   & LF & HT &
                   "rdrand %0"            & LF & HT &
                   "jc 2f"                & LF & HT &
                   "loop 1b"              & LF & HT &
                   "cmovncl %%ecx, %%eax" & LF & HT &
                   "2:",
       Outputs  => rx'Asm_Output ("=a", r),
       Inputs   => rlim'Asm_Input ("n", RL),
       Clobber  => "rcx, cc",
       Volatile => True);
    return r;
  end rand;

  function seed return sx with SPARK_Mode => Off is
    r : sx;  -- return value
  begin
    System.Machine_Code.Asm  -- x86 att
      (Template => "xorl %%eax, %%eax"    & LF & HT &
                   "movl %1, %%ecx"       & LF & HT &
                   "1:"                   & LF & HT &
                   "rdseed %0"            & LF & HT &
                   "jc 2f"                & LF & HT &
                   "loop 1b"              & LF & HT &
                   "cmovncl %%ecx, %%eax" & LF & HT &
                   "2:",
       Outputs  => sx'Asm_Output ("=a", r),
       Inputs   => rlim'Asm_Input ("n", RL),
       Clobber  => "rcx, cc",
       Volatile => True);
    return r;
  end seed;

end rng;
