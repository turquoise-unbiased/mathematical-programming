## opt_uninte.R [4.4.1] 2024, Wojciech Lawren, All rights reserved.
## Optimizing [p.r.] moment / interval | collision of pseudo-random integers of uniform distribution
library(parallel)

## mode : numeric | typeof : double | str : array
l <- 1e2 ; a <- 1e2 ; b <- 1e5  # random generator parameters
n <- 3L  # trials n ^ 3 * k
k <- n   # array dim 3

S.show <- T  # print score on terminal
S.save <- F  # save score to file

cc <- detectCores()    # number of CPU cores >=1 nodes involved in simulation
ff <- cumprod(seq(n))  # cumulative product vector 1:n series coefficient

p.switch = function(i) {
  switch(i[1],
    rep(ff * sample(ff * l, 1), times = n ^ 2),       # n ^ 3
    rep(ff * sample(ff * a, 1), each  = n, times = n),
    rep(ff * sample(ff * b, 1), each  = n ^ 2))       # opt <- ff * l|a|b
}

parameters = function() {
  cl = makeForkCluster(min(k * 3, cc))                # cluster of processes
  on.exit(stopCluster(cl))

  P <- array(rep(seq(3), times = k), dim = c(1, 3, k)) # parameters array
  parApply(cl, P, c(2, 3), function(i) p.switch(i))   # parallel apply
}

orbit = function(r) {
  n.l <- r[1] ; a <- r[2] ; b <- r[3]

  ran.v <- round(runif(n.l, a, b))                    # random integers vector
  v.dif <- abs(diff(ran.v, lag = 1))                  # neighbors distance vector |i,j|

  d.men <- mean(v.dif)                                # mean distance
  d.med <- median(v.dif)                              # median distance
  d.mad <- mad(v.dif, center = d.med, constant = 1)   # median absolute deviation of distance
  ran.c <- length(ran.v) - length(unique(ran.v))      # integers collision
  dif.c <- length(v.dif) - length(unique(v.dif))      # distance collision

  round(c(b - a, n.l, d.men, d.med, d.mad, ran.c, dif.c))
}

S.col <- factor(c("intvl", "n.rand", "dist.men", "dist.med", "dist.mad", "coll.rand", "coll.dist"))

simulate = function(P) {
  cl = makeForkCluster(min(n ^ 3 * k, cc))            # cluster of processes
  on.exit(stopCluster(cl))

  parApply(cl, P, c(1, 3), function(r) orbit(r))      # parallel apply
}

S <- aperm(simulate(parameters()), perm = c(2, 1, 3)) ; colnames(S) <- S.col  # score array

if (S.show) { for (i in seq(k)) { print(summary(S[,,i])) ; print(cor(S[,,i])) }}
if (S.save) { write.csv(S, "data.csv") }
