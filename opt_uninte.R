## opt_uninte.R [4.4.1] 2024, Wojciech Lawren, All rights reserved.
## Optimizing [p.r.] moment / interval | collision of pseudo-random integers of uniform distribution
library(parallel)

## mode : numeric | typeof : double | str : array
a1 <- 1e2 ; b1 <- 1e5 ; l1 <- 1e2  # random generator parameters
a2 <- 1e3 ; b2 <- 1e5 ; l2 <- 1e3  # random generator parameters
k  <- 2  # rng sets

n    <- 3  # trials n ^ 3 * k
show <- T  # print score on terminal
save <- F  # save score to file

cc <- detectCores()    # number of CPU cores >=1 nodes involved in simulation
ff <- cumprod(seq(n))  # cumulative product vector 1:n series coefficient

pc = function(i) {
  switch(i[1],
    rep(ff * l1, times = n ^ 2),                 # n ^ 3
    rep(ff * a1, each  = n, times = n),
    rep(ff * b1, each  = n ^ 2),
    rep(ff * l2, times = n ^ 2),                 # n ^ 3
    rep(ff * a2, each  = n, times = n),
    rep(ff * b2, each  = n ^ 2))
}

parameters = function() {
  cl = makeForkCluster(min(k * 3, cc))           # cluster of processes
  on.exit(stopCluster(cl))

  P <- array(seq(k * 3), dim = c(1, 3, k))       # parameter array
  parApply(cl, P, c(2, 3), function(i) pc(i))    # parallel apply
}

orbit = function(r) {
  l <- r[1] ; a <- r[2] ; b <- r[3]

  vec <- round(runif(l, a, b))                   # random integers vector
  meq <- abs(diff(vec, lag = 1))                 # neighbors distance vector |i,j|

  meq_men <- mean(meq)                           # mean distance
  meq_med <- median(meq)                         # median distance
  meq_mad <- mad(meq, center = meq_med, constant = 1) # median absolute deviation of distance
  vec_col <- length(vec) - length(unique(vec))   # integers collision
  meq_col <- length(meq) - length(unique(meq))   # distance collision

  round(c(b - a, l, meq_men, meq_med, meq_mad, vec_col, meq_col))
}

col.f <- factor(c("intvl", "n.rand", "dist.men", "dist.med", "dist.mad", "coll.rand", "coll.dist"))

simulate = function(P) {
  cl = makeForkCluster(min(n ^ 3 * k, cc))       # cluster of processes
  on.exit(stopCluster(cl))

  parApply(cl, P, c(1, 3), function(r) orbit(r)) # parallel apply
}

S <- aperm(simulate(parameters()), perm = c(2, 1, 3)) ; colnames(S) <- col.f  # score array

if (show) { for (i in seq(k)) { print(summary(S[,,i])) ; print(cor(S[,,i])) }}
if (save) { write.csv(S, "data.csv") }
