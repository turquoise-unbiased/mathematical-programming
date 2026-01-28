--  2020, Wojciech Lawren, All rights reserved.
--  SPARK interface for x86 RNG (Cryptographic Co-Processor).
--  [GNAT 14.1.0]
pragma Ada_2022;

with System.Machine_Code; use System.Machine_Code;

--  RDRAND RDSEED
package body rng with
   SPARK_Mode is

   RL : constant Positive := 2;  --  retry limit
   use ASCII;  --  LF HT

   function rand return rx is
      r : rx;  --  return value
   begin
      Asm
        (Template => "xorl %%eax, %%eax"    & LF & HT &
                     "movl %1, %%ecx"       & LF & HT &
                     "1:"                   & LF & HT &
                     "rdrand %0"            & LF & HT &
                     "jc 2f"                & LF & HT &
                     "loop 1b"              & LF & HT &
                     "cmovncl %%ecx, %%eax" & LF & HT &
                     "2:",
         Outputs  => (rx'Asm_Output ("=a", r)),
         Inputs   => (Positive'Asm_Input ("n", RL)),
         Clobber  => "rcx, cc",
         Volatile => True);
      return r;
   end rand;

   function seed return sx is
      r : sx;  --  return value
   begin
      Asm
        (Template => "xorl %%eax, %%eax"    & LF & HT &
                     "movl %1, %%ecx"       & LF & HT &
                     "1:"                   & LF & HT &
                     "rdseed %0"            & LF & HT &
                     "jc 2f"                & LF & HT &
                     "loop 1b"              & LF & HT &
                     "cmovncl %%ecx, %%eax" & LF & HT &
                     "2:",
         Outputs  => (sx'Asm_Output ("=a", r)),
         Inputs   => (Positive'Asm_Input ("n", RL)),
         Clobber  => "rcx, cc",
         Volatile => True);
      return r;
   end seed;

end rng;
