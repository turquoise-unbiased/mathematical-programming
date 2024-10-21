## opt_uninte.R [4.4.1] 2024, Wojciech Lawren, All rights reserved.
## Optimizing [p.r.] moment / interval | collision of pseudo-random integers of uniform distribution
library(parallel)
cc <- parallel::detectCores()  # number of CPU cores >=1 nodes involved in simulation

source("cls_uninte.R")
f <- new("Fun")  # object class Fun

parameters <- function() {
  cl <- parallel::makeForkCluster(min(3L * f@k, cc))                       # cluster of processes
  on.exit(parallel::stopCluster(cl))
  parallel::parApply(cl, f@A, c(2L, 3L), function(r) f@.switch(r, f))    # parallel apply
}

simulate <- function(A) {
  cl <- parallel::makeForkCluster(min(f@t ^ 3L * f@k, cc))                 # cluster of processes
  on.exit(parallel::stopCluster(cl))
  parallel::parApply(cl, A, c(1L, 3L), function(r) f@orbit(r))            # parallel apply
}

S <- aperm(simulate(parameters()), perm = c(2L, 1L, 3L)) ; colnames(S) <- f@col  # score array

if (f@show) { for (i in seq(f@k)) { print(summary(S[,,i])) ; print(cor(S[,,i])) }}
if (f@save) { write.csv(S, f@csv) }
