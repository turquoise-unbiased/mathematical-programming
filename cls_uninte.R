## cls_uninte.R [4.4.1] 2024, Wojciech Lawren, All rights reserved.
## Optimizing [p.r.] moment / interval | collision of pseudo-random integers of uniform distribution

## Classes ##

setClass("UN",  contains  = c("VIRTUAL"), sealed = T)

setClass("Log", contains  = c("UN"), sealed = T,
                slots     = list(show = "logical",         # print score on terminal
                                 save = "logical",         # save score to file
                                 csv  = "character"),      # csv file to save
                prototype = list(show = T,
                                 save = F,
                                 csv  = "data.csv"),
                validity  = function(object) {})

setClass("Col", contains  = c("UN"), sealed = T,
                slots     = list(col = "factor"),          # score array colnames
                prototype = list(col = factor(c("intvl", "n.rand", "dist.men", "dist.med", "dist.mad",
                                                "coll.rand", "coll.dist"))),
                validity  = function(object) {})

setClass("Lab", contains  = c("Log", "Col"), sealed = T,
                slots     = list(n = "numeric",            # runif n
                                 a = "numeric",            # runif min
                                 b = "numeric",            # runif max
                                 t = "numeric",            # trials t ^ 3 * k
                                 k = "numeric"),           # score array dim 3
                prototype = list(n = 1e+04,
                                 a = 1e+03,
                                 b = 1e+07,
                                 t = 3L,
                                 k = 2L),
                validity  = function(object) {})

setClass("Sim", contains  = c("Lab"), sealed = T,
                slots     = list(A  = "call",            # parameters array call
                                 ff = "numeric"),            # cumulative product vector 1:t series coefficients
                validity  = function(object) {})

## Methods ##

setMethod("initialize", sealed = T, signature = c("Sim"),                                  # initialize Sim slots
          definition = function(.Object, ...) {
            .Object    <- callNextMethod(.Object, ...)
            .Object@A  <- substitute(array(data = rep(seq(3L), times = k),
                                           dim  = c(1L, 3L, k)), list(k = .Object@k))
            .Object@ff <- cumprod(seq(.Object@t))
            .Object
          })

## Functions ##

v.switch <- as.function(alist(r =, {                                           # switch parameters sets
  switch(r,
    rep((m@ff * rbinom(1L, m@n, runif(1L, 2e-01))), times = m@t ^ 2L),              # t ^ 3
    rep((m@ff * rbinom(1L, m@a, runif(1L, 2e-01))), each  = m@t, times = m@t),
    rep((m@ff * rbinom(1L, m@b, runif(1L, 2e-01))), each  = m@t ^ 2L))
}))

orbit <- as.function(alist(r =, {                                              # compute runif stat trial
  names(r) <- names(formals(runif))
  formals(runif) <- ( r <- as.list(r) )

  ran.v <- floor(runif())                              # random integers vector
  v.dif <- abs(diff(ran.v, lag = 1L, differences = 1L))                         # neighbors distance vector

  c((r$max - r$min),
    r$n,
    mean(v.dif),                                               # mean distance
    ( d.med <- median(v.dif) ),                                             # median distance
    mad(v.dif, center = d.med, constant = 1L),                # median absolute deviation of distance
    sum(duplicated(ran.v)),                     # integers collision
    sum(duplicated(v.dif)))                     # distance collision
}))
