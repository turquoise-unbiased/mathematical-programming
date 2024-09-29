## opt_uninte.R [4.1.2] 2024, Wojciech Lawren, All rights reserved.
## Optimizing [p.r.] interval | collision of pseudo-random integers of uniform distribution on interval

low <- 10 ^ 2 ; high <- 10 ^ 5 ; size <- 10 ^ 2  # random generator parameters [low:high].size
LOW <- low    ; HIGH <- high   ; SIZE <- size    # initial random generator parameters [LOW:HIGH].SIZE

LOOP     <- 3      # for seq(LOOP) ^ 3
PRINT_MS <- TRUE   # print score on terminal
SAVE_MS  <- FALSE  # save score to file

orbit = function(size, low, high) {
  vec <- round(runif(size, low, high))           # random integers vector
  meq <- abs(diff(vec, lag = 1))                 # neighbors distance vector |i,j|

  meq_men <- mean(meq)                           # mean distance
  meq_med <- median(meq)                         # median distance
  meq_mad <- mad(meq, center = meq_med)          # median absolute deviation of distance
  vec_col <- length(vec) - length(unique(vec))   # integers collision
  meq_col <- length(meq) - length(unique(meq))   # distance collision

  round(c(low, high, size, meq_men, meq_med, meq_mad, vec_col, meq_col))
}

ms <- matrix(nrow = LOOP ^ 3, ncol = 8)  # score matrix
colnames(ms) <- c("low", "high", "size", "dist.men", "dist.med", "dist.mad", "coll.rand", "coll.dist")
idx <- 1  # index

for (i in seq(LOOP)) {
  high <- high * i
  low  <- LOW
  for (j in seq(LOOP)) {
    low  <- low * j
    size <- SIZE
    for (k in seq(LOOP)) {
      size <- size * k
      ms[idx,] <- orbit(size, low, high)
      idx <- idx + 1
}}}

if (PRINT_MS) { print(summary(ms)) ; print(cor(ms)) }
if (SAVE_MS)  { write.csv(ms, "data.csv") }
