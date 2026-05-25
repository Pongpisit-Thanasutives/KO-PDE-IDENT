gc()
rm(list = ls())

library(reticulate)
use_condaenv("base")
np <- import("numpy")

library(selectiveInference)

X_path <- "../Cache/X_pre_kdv_noise50_sample2000.npy"
y_path <- "../Cache/y_pre_kdv_noise50_sample2000.npy"
X_pre <- np$load(X_path)
y_pre <- np$load(y_path)
X_pre <- scale(X_pre, scale=TRUE)
y_pre <- scale(y_pre, scale=FALSE)
n <- nrow(y_pre)

alpha = 0.10 # 0.05, 0.10
# run forward stepwise
fsfit = fs(X_pre,y_pre,intercept=FALSE,normalize=FALSE)
# run sequential inference with estimated sigma
# out_fs = fsInf(fsfit,alpha=alpha,type="aic",mult=log(n)) # # _screening
out_fs = fsInf(fsfit,alpha=alpha)

save_file = "./R_data/fsInf_active_kdv_noise50_sample2000_alpha10.rds"
saveRDS(out_fs, file = save_file)
readRDS(save_file)
