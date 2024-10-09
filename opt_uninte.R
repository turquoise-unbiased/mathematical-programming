## opt_uninte.R [4.1.2] 2024, Wojciech Lawren, All rights reserved.
## Optimizing [p.r.] interval | collision of pseudo-random integers of uniform distribution on interval
library(parallel)

## mode : numeric | typeof : double | str : vector matrix
LOW <- 1e2 ; HIGH <- 1e5 ; SIZE <- 1e2  # initial random generator parameters [LOW:HIGH].SIZE

n    <- 3  # trials n ^ 3
SHOW <- T  # print score on terminal
SAVE <- F  # save score to file

ff <- cumprod(seq(n))  # cumulative product vector 1:n series coefficient

pc = function(i) {
  switch(i[1],
    rep(ff * SIZE, n ^ 2),                       # n*n^2
    rep(sequence(rep(n, n), ff * LOW, 0), n),    # n*n*n
    sequence(rep(n ^ 2, n), ff * HIGH, 0))       # n^2*n
}

parameters = function() {
  cl = makeForkCluster(3)                        # cluster of processes
  on.exit(stopCluster(cl))

  P <- matrix(seq(3), nrow = 1, ncol = 3)        # parameter matrix
  parApply(cl, P, 2, function(i) pc(i))          # parallel apply
}

orbit = function(r) {
  size <- r[1] ; low <- r[2] ; high <- r[3]

  vec <- round(runif(size, low, high))           # random integers vector
  meq <- abs(diff(vec, lag = 1))                 # neighbors distance vector |i,j|

  meq_men <- mean(meq)                           # mean distance
  meq_med <- median(meq)                         # median distance
  meq_mad <- mad(meq, center = meq_med)          # median absolute deviation of distance
  vec_col <- length(vec) - length(unique(vec))   # integers collision
  meq_col <- length(meq) - length(unique(meq))   # distance collision

  round(c(low, high, size, meq_men, meq_med, meq_mad, vec_col, meq_col))
}

simulate = function(P) {
  cl = makeForkCluster(detectCores())            # cluster of processes
  on.exit(stopCluster(cl))

  parApply(cl, P, 1, function(r) orbit(r))       # parallel apply
}

S <- t(simulate(parameters()))                   # score matrix
colnames(S) <- c("low", "high", "size", "dist.men", "dist.med", "dist.mad", "coll.rand", "coll.dist")

if (SHOW) { print(summary(S)) ; print(cor(S)) }
if (SAVE) { write.csv(S, "data.csv") }
