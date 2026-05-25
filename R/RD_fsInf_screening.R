gc()
rm(list = ls())

library(reticulate)
use_condaenv("base")
np <- import("numpy")

library(selectiveInference)

X_pre <- np$load("../Cache/X_pre_RD_2025.npy")
y_pre <- np$load("../Cache/y_pre_RD_2025.npy")
y_pre <- y_pre[, 2, drop=FALSE]
X_pre <- scale(X_pre, scale=TRUE)
y_pre <- scale(y_pre, scale=FALSE)

alpha = 0.10 # 0.05, 0.10
# run forward stepwise
fsfit = fs(X_pre,y_pre,intercept=FALSE,normalize=FALSE)
# run sequential inference with estimated sigma
# out_fs = fsInf(fsfit,alpha=alpha,type="aic",mult=log(nrow(y_pre))) # _screening
out_fs = fsInf(fsfit,alpha=alpha)

save_file = "./R_data/fsInf_active_RD_v_sample10000_alpha10.rds"
saveRDS(out_fs, file = save_file)
readRDS(save_file)
