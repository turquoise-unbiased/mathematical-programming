## opt_uninte.R [4.4.1] 2024, Wojciech Lawren, All rights reserved.
## Optimizing [p.r.] moment / interval | collision of pseudo-random integers of uniform distribution
library("parallel")
.cc <- parallel::detectCores()  # number of CPU cores >=1 nodes involved in simulation

source("cls_uninte.R")
assign("m", new("Sim"), envir = ( e.sim <- new.env() ))  # str class Sim

parameters <- function() {
  nodes <- min((3L * m@k), .cc)
  cl <- parallel::makeForkCluster(nodes)                       # cluster of processes
  on.exit(parallel::stopCluster(cl))
  parallel::clusterSetRNGStream(cl, iseed = 37L)               # distribute CMRG streams
  parallel::parApply(cl, eval(m@A), c(2L, 3L), function(r) v.switch(r))      # parallel apply
}

simulate <- function(P) {
  nodes <- min((m@t ^ 3L * m@k), .cc)
  cl <- parallel::makeForkCluster(nodes)                 # cluster of processes
  on.exit(parallel::stopCluster(cl))
  parallel::clusterSetRNGStream(cl, iseed = 59L)         # distribute CMRG streams
  parallel::parApply(cl, P, c(1L, 3L), function(r) orbit(r))              # parallel apply
}

attach(e.sim, pos = 2L)
S <- aperm(simulate(parameters()), perm = c(2L, 1L, 3L))  # score array

colnames(S) <- m@col
if (m@show) { for (i in seq(m@k)) { print(summary(S[,,i])) ; print(cor(S[,,i])) }}
if (m@save) { write.csv(S, m@csv) }
