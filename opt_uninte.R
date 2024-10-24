## opt_uninte.R [4.4.1] 2024, Wojciech Lawren, All rights reserved.
## Optimizing [p.r.] moment / interval | collision of pseudo-random integers of uniform distribution
library("parallel")
.cc <- parallel::detectCores()  # number of CPU cores >=1 nodes involved in simulation

source("cls_uninte.R")
assign("m", new("Sim"), envir = ( e.sim <- new.env() ))  # object class Sim

parameters <- function() {
  cl <- parallel::makeForkCluster(min((3L * m@k), .cc))                       # cluster of processes
  on.exit(parallel::stopCluster(cl))
  # parallel::clusterEvalQ(cl, attach(e.sim, pos = 2L))
  parallel::parApply(cl, eval(m@A), c(2L, 3L), function(r) v.switch(r))      # parallel apply
}

simulate <- function(P) {
  cl <- parallel::makeForkCluster(min((m@t ^ 3L * m@k), .cc))                 # cluster of processes
  on.exit(parallel::stopCluster(cl))
  parallel::parApply(cl, P, c(1L, 3L), function(r) orbit(r))              # parallel apply
}

attach(e.sim, pos = 2L)
S <- aperm(simulate(parameters()), perm = c(2L, 1L, 3L))  # score array

colnames(S) <- m@col
if (m@show) { for (i in seq(m@k)) { print(summary(S[,,i])) ; print(cor(S[,,i])) }}
if (m@save) { write.csv(S, m@csv) }
