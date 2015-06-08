setwd("C:\\Users\\Stephen\\Google Drive\\PhD\\Dissertation\\3. network analysis\\data")
library(plyr)
library(reshape2)
library(ggplot2)
library(igraph)
library(lme4)
library(MASS)
library(memisc)

dfp <- read.table("reg_df_qty_rev_demog_pca.csv",sep=",",header=T)
dfw <- read.table("Wlong.csv",sep=",",header=T)
df <- merge(dfw,dfp,by=c('mem_no','pref'))
# df <- read.table("dfregall.csv", sep=",", header=T)
# df <- unique(df)
names(df)

# rescale
dfrs <- df
cols <- c("netWeight","age","revPC0", "revPC1","revPC2",
          "qtyPC0","qtyPC1")
dfrs[,cols] <- scale(dfrs[,cols],center=F,scale=T)


# # regular GLM
# g0 <- glm(qty ~ gender + marriage + age,
#              family=poisson(link="log"),
#              data=df)
# summary(g0)
#
# g1 <- glm(qty ~ gender + marriage + age+
#               qtyPC0 + qtyPC1 + qtyPC2 + qtyPC3 +
#               revPC0 + revPC1 + revPC2 + revPC3 ,
#             family=poisson(link="log"),
#             data=df)
# summary(g1)
#
# g2 <- glm(qty ~ gender + marriage + age+
#             qtyPC0 + qtyPC2 + qtyPC3 +
#             revPC0 + revPC1 + revPC3 ,
#           family=poisson(link="log"),
#           data=df)
# summary(g2)
#
# mtable(g0,g1,g2)


# MIxed Effects Modles

me1a <- glmer(qty ~ gender + marriage +
               (1 + age|mem_no),
             family=poisson(link="log"),
             data=df)
summary(me1a)

me1b <- glmer(qty ~ gender + marriage +
               (1 + age| pref),
             family=poisson(link="log"),
             data=df)
summary(me1b)




me2a <- glmer(qty ~ gender + marriage + age +
               qtyPC0 + qtyPC1 +
               revPC0 + revPC1 + revPC2 +
               (1  | pref),
             family=poisson(link="log"),
             data=df)
summary(me2a)



me3a <- glmer(qty ~ gender + marriage + age +
                qtyPC0 + qtyPC1 +
                revPC0 + revPC1 + revPC2 +
                netWeight +
                (1  | pref),
              family=poisson(link="log"),
              data=df)
summary(me3a)


anova(me2a,me3a)



dput("cconma_netweight_poisson_mixeff.RData")






n <- 30
prefs <- unique(df$pref)
modlist <- list()
zvec <- rep(NA,length(prefs))
for ( i in 1:length(prefs)) {
  pref_i <- prefs[i]
  data <- df[which(df$pref==pref_i),]
  if ( dim(data)[1]>=n ) {
       holder <- glm(qty ~ gender + marriage + age +
                       qtyPC0 + qtyPC1 +
                       revPC0 + revPC1 + revPC2 +
                        netWeight,
                     family=poisson(link='log'),
                      data = data)
         modlist[[length(modlist)+1]] <- holder
         zvec[i] <- summary(holder)$coef[10,3]
#     #ERROR HANDLING
#     possibleError <- tryCatch(
#       summary(modlist[[i]])$coef['z value']['netWeight'],
#       error=function(e) e
#     )
#     if(inherits(possibleError, "error")) next
#     #REAL WORK
#     tvec[i] <- summary(modlist[[i]])$coef['z value']['netWeight']
#     error=function(e) next
#   } else {
#     #modlist[[i]] <- NA
#   }
  }

}


#-------------------------------------------------
# Check significant of netWeight on individual product category regressions
z <- na.omit(unlist(zvec))
q95 <- qnorm(.975, 0, 1)
q99 <- qnorm(.995, 0, 1)

prop95 <- length(z[abs(z)>q95]) / length(z)
prop99 <- length(z[abs(z)>q99]) / length(z)

print(paste0(length(z)," products with >= ", n, " observations"))
print(paste0(round(prop95,3)*100, "% significant at alpha=0.05"))
print(paste0(round(prop99,3)*100, "% significant at alpha=0.01"))










