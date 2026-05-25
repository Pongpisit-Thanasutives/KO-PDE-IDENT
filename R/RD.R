gc()
rm(list = ls())

library(R.matlab)
library(reticulate)
np <- import("numpy")

library(leaps)
library(selectiveInference)
library(knockoff)
library(cvCovEst)

# 1 for u | 2 for v
X_pre <- np$load("../Cache/X_pre_RD_2025.npy")
y_pre <- np$load("../Cache/y_pre_RD_2025.npy")
y_pre <- y_pre[, 2, drop=FALSE]
n = nrow(y_pre)

### data preprocessing ###
X_pre <- scale(X_pre)
y_pre <- scale(y_pre)

nvmax = 7
bestsubsets = regsubsets(y_pre ~ ., data=data.frame(X_pre), nvmax=nvmax, intercept=FALSE, method="exhaustive")
bestsubsets = summary(bestsubsets)
selected = as.numeric(which(bestsubsets$which[nvmax,]))

X_sel <- X_pre[, selected]
X_not_sel <- X_pre[, -selected]
model <- lm(y_pre ~ . + 0, data = data.frame(X_sel))
y_est <- X_sel %*% model$coefficients
X_test <- cbind(y_est, X_not_sel)

alpha = 0.01
sigmahat = 0.001*estimateSigma(X_test,y_pre,intercept=FALSE)$sigmahat
sigmahat = sd(y_pre)
# run forward stepwise
fsfit = fs(X_test,y_pre,intercept=FALSE,normalize=FALSE)
# run sequential inference with estimated sigma
out_fs = fsInf(fsfit,sigma=sigmahat,alpha=alpha,type="active")
fs_indices = out_fs$vars[1:out_fs$khat]
print(fs_indices)
print(forwardStop(out_fs$pv, alpha=alpha)) # max alpha

steps <- c()
alphas <- seq(0.01, 0.1, by = 0.01)
for (alpha in alphas) {
    steps <- c(steps, forwardStop(out_fs$pv, alpha=alpha))
}
plot(alphas, steps, yaxt="n", xlab="alpha")
axis(2, at = unique(steps))

save_file = "./R_data/fsInf_RD_v.rds"
saveRDS(out_fs, file = save_file)
readRDS(save_file)

