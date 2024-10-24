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
                slots     = list(n = "integer",            # runif n
                                 a = "integer",            # runif min
                                 b = "integer",            # runif max
                                 t = "integer",            # trials t ^ 3 * k
                                 k = "integer"),           # score array dim 3
                prototype = list(n = as(1e3, "integer"),
                                 a = as(1e3, "integer"),
                                 b = as(1e6, "integer"),
                                 t = 3L,
                                 k = 2L),
                validity  = function(object) {})

Sim <- setClass("Sim",
                contains  = c("Lab"),
                slots     = list(ff = "vector",            # cumulative product vector 1:t series coefficients
                                 A  = "array"),            # parameters array
                validity  = function(object) {})

## Methods ##

setMethod("initialize", signature = c("Sim"),                                  # initialize Sim slots
          definition = function(.Object, ...) {
            .Object    <- callNextMethod(.Object, ...)
            .Object@ff <- cumprod(seq(.Object@t))
            .Object@A  <- array(rep(seq(3L), times = .Object@k), dim = c(1L, 3L, .Object@k))
            .Object
          })

setGeneric("v.switch", def = function(r, m) standardGeneric("v.switch"))       # switch parameters sets
setMethod("v.switch", signature = c("numeric", "Sim"),
          definition = function(r, m) {
            switch(r[1L],
              rep(m@ff * rbinom(1L, m@n, runif(1L, 2e-1)), times = m@t ^ 2L),              # t ^ 3
              rep(m@ff * rbinom(1L, m@a, runif(1L, 2e-1)), each  = m@t, times = m@t),
              rep(m@ff * rbinom(1L, m@b, runif(1L, 2e-1)), each  = m@t ^ 2L))
          })

setMethod("v.switch", signature = c("numeric"),
          definition = function(r) {
            switch(r[1L],
              rep(m@ff * rbinom(1L, m@n, runif(1L, 2e-1)), times = m@t ^ 2L),              # t ^ 3
              rep(m@ff * rbinom(1L, m@a, runif(1L, 2e-1)), each  = m@t, times = m@t),
              rep(m@ff * rbinom(1L, m@b, runif(1L, 2e-1)), each  = m@t ^ 2L))
          })

setGeneric("orbit", def = function(r) standardGeneric("orbit"))                # compute runif stat trial
setMethod("orbit", signature = c("numeric"),
          definition = function(r) {
            r.n <- r[1L] ; r.a <- r[2L] ; r.b <- r[3L]

            ran.v <- round(runif(r.n, r.a, r.b), 0L)                               # random integers vector
            v.dif <- abs(diff(ran.v, lag = 1L))                                # neighbors distance vector |i,j|

            d.men <- mean(v.dif)                                               # mean distance
            d.med <- median(v.dif)                                             # median distance
            d.mad <- mad(v.dif, center = d.med, constant = 1L)                 # median absolute deviation of distance
            ran.c <- length(ran.v) - length(unique(ran.v))                     # integers collision
            dif.c <- length(v.dif) - length(unique(v.dif))                     # distance collision

            round(c(r.b - r.a, r.n, d.men, d.med, d.mad, ran.c, dif.c), 3L)
          })
