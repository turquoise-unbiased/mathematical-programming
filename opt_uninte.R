## opt_uninte.R [4.4.1] 2024, Wojciech Lawren, All rights reserved.
## Optimizing [p.r.] moment / interval | collision of pseudo-random integers of uniform distribution
library(parallel)
cc <- detectCores()  # number of CPU cores >=1 nodes involved in simulation

source("cls_uninte.R")
m <- new("Sim")  # object class Sim

parameters = function() {
  cl = makeForkCluster(min(3L * m@k, cc))                       # cluster of processes
  on.exit(stopCluster(cl))
  parApply(cl, m@P, c(2L, 3L), function(i) p.switch(i, m))      # parallel apply
}

simulate = function(P) {
  cl = makeForkCluster(min(m@t ^ 3L * m@k, cc))                 # cluster of processes
  on.exit(stopCluster(cl))
  parApply(cl, P, c(1L, 3L), function(r) orbit(r))              # parallel apply
}

S <- aperm(simulate(parameters()), perm = c(2L, 1L, 3L)) ; colnames(S) <- m@col  # score array

if (m@show) { for (i in seq(m@k)) { print(summary(S[,,i])) ; print(cor(S[,,i])) }}
if (m@save) { write.csv(S, m@csv) }
