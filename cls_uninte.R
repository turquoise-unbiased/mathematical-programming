## cls_uninte.R [4.4.1] 2024, Wojciech Lawren, All rights reserved.
## Optimizing [p.r.] moment / interval | collision of pseudo-random integers of uniform distribution

## Classes ##

UN  <- setClass("UN", contains = c("VIRTUAL"))

Log <- setClass("Log",
                contains  = c("UN"),
                slots     = list(show = "logical",         # print score on terminal
                                 save = "logical",         # save score to file
                                 csv  = "character"),      # csv file to save
                prototype = list(show = T,
                                 save = F,
                                 csv  = "data.csv"),
                validity  = function(object) {})

Col <- setClass("Col",
                contains  = c("UN"),
                slots     = list(col = "factor"),          # score array colnames
                prototype = list(col = factor(c("intvl", "n.rand", "dist.men", "dist.med", "dist.mad",
                                                "coll.rand", "coll.dist"))),
                validity  = function(object) {})

Lab <- setClass("Lab",
                contains  = c("Log", "Col"),
                slots     = list(n = "numeric",            # runif n
                                 a = "numeric",            # runif min
                                 b = "numeric",            # runif max
                                 t = "numeric",            # trials t ^ 3 * k
                                 k = "numeric"),           # score array dim 3
                prototype = list(n = 1e2,
                                 a = 1e2,
                                 b = 1e5,
                                 t = 3L,
                                 k = 2L),
                validity  = function(object) {})

Sim <- setClass("Sim",
                contains  = c("Lab"),
                slots     = list(ff = "vector",            # cumulative product vector 1:t series coefficients
                                 M  = "matrix",            # ff * n|a|b
                                 P  = "array"),            # parameters array
                validity  = function(object) {})

## Methods ##

setMethod("initialize", signature = c("Sim"),                                  # initialize Sim slots
          definition = function(.Object, ...) {
            .Object    <- callNextMethod(.Object, ...)
            .Object@ff <- cumprod(seq(.Object@t))
            .Object@M  <- matrix(c(.Object@ff * .Object@n, .Object@ff * .Object@a,
              .Object@ff * .Object@b), ncol = 3L)
            .Object@P  <- array(rep(seq(3L), times = .Object@k), dim = c(1L, 3L, .Object@k))
            .Object
          })

setGeneric("p.switch", def = function(i, m) standardGeneric("p.switch"))       # switch parameters sets
setMethod("p.switch", signature = c("numeric", "Sim"),
          definition = function(i, m) {
            switch(i[1L],
              rep(m@ff * sample(m@M[,1L], 1L), times = m@t ^ 2L),              # t ^ 3
              rep(m@ff * sample(m@M[,2L], 1L), each  = m@t, times = m@t),
              rep(m@ff * sample(m@M[,3L], 1L), each  = m@t ^ 2L))
          })

setGeneric("orbit", def = function(r) standardGeneric("orbit"))                # compute runif stat trial
setMethod("orbit", signature = c("numeric"),
          definition = function(r) {
            r.n <- r[1L] ; r.a <- r[2L] ; r.b <- r[3L]

            ran.v <- round(runif(r.n, r.a, r.b))                               # random integers vector
            v.dif <- abs(diff(ran.v, lag = 1L))                                # neighbors distance vector |i,j|

            d.men <- mean(v.dif)                                               # mean distance
            d.med <- median(v.dif)                                             # median distance
            d.mad <- mad(v.dif, center = d.med, constant = 1L)                 # median absolute deviation of distance
            ran.c <- length(ran.v) - length(unique(ran.v))                     # integers collision
            dif.c <- length(v.dif) - length(unique(v.dif))                     # distance collision

            round(c(r.b - r.a, r.n, d.men, d.med, d.mad, ran.c, dif.c))
          })
