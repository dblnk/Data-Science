#####################################################################
### Development of a machine learning model aiming to predict the ###
### presence of a brain tumor in brain MRI scan images            ###
#####################################################################

# you can clone the project from my github repository: 
# ""

# Or you build it quickly yourself by setting the working directory to a directory with "data", "fig" and "model" sub-directories present, then:

# If you are registered and logged into Kaggle, then download the archived data set from this URL into your data sub-directory :
# https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor/download?datasetVersionNumber=3

#then extract the archive with this code
dl <- "data/archive.zip"
brain_tumor <- "data/brain-tumor/Brain Tumor.csv"
if(!file.exists(brain_tumor))
  unzip(dl, exdir="data/brain-tumor")

# Alternatively, use this code to download the data from my github repository:

dl <- "data/brain-tumor.zip"
if(!file.exists(dl))
  download.file("githubusercontent.com/...", dl)
brain_tumor <- "data/brain-tumor/Brain Tumor.csv"
if(!file.exists(brain_tumor))
  unzip(dl, exdir="data/brain-tumor")

# the data set was derived by the kaggle submitter JAKESH BOHAJU from https://www.smir.ch/BRATS/Start2015#!#evaluation . The paper describing the BRATS brain segmentation challenge is available under https://ieeexplore.ieee.org/document/6975210/ by Menze et al., The Multimodal Brain TumorImage Segmentation Benchmark (BRATS), IEEE Trans. Med. Imaging, 2015. 

##### After succesfully unzipping the file, change the structure of the subdirectories in the terminal using this unix code:
# cd data/brain-tumor/Brain Tumor/Brain Tumor
# pwd
# mv * ../
# cd ../
# rmdir "Brain Tumor"

##### load required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm")
if(!require(gam)) install.packages("gam")
if(!require(jpeg)) install.packages("jpeg", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(purrr)) install.packages("purrr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(ggrepel)) install.packages("ggrepel", repos = "http://cran.us.r-project.org")

# make numbers more readable
options(digits=7)
bt <- read.csv2(file=brain_tumor, sep=",", dec=".")
bt[,3:14] <- signif(bt[,3:14], digits=5)
bt[,15] <- round(bt[,15], digits=3)
str(bt)
names(bt)
# Simplify names
bt <- bt %>% rename(StdDev = Standard.Deviation)

# check value structure
summary(bt)
# check frequency of the outcome
prop.table(table(bt$Class))

# Check if any of the predictors have very low variance
nearZeroVar(bt[,2:15], saveMetrics=TRUE)

# Remove Coarseness due to low Variance.
bt <- bt %>% select(-Coarseness)

ds_theme_set()
#Let's look at the value distributions
bp <- bt %>% pivot_longer(Mean:Correlation, names_to="parameter", values_to="value") %>% 
  ggplot(aes(parameter, value, color=factor(Class, labels=c("normal","tumor")))) + 
  geom_boxplot(outlier.size=0.2) +
  scale_color_manual(values = c("#2e8b57", "#F8766D"))+
  guides(color=guide_legend(title=NULL))+
  scale_y_continuous(trans="log10")+
  ggtitle("Distribution of features between outcomes")+
  theme(plot.title = element_text(hjust=0.5, size=12))+
  theme(axis.text.x = element_text(angle=90, hjust=1.0))
bp
ggsave("fig/boxplot_prd_outc.pdf")

# Exploring the correlations between the variables
cor(bt[,2:14]) # we observe a wide range of correlation values, including highly correlated and anti-correlated ones

# Creating a validation data set: the final holdout set
set.seed(1996)
test_index <- createDataPartition(bt$Class, times=1, p=0.1, list=FALSE)
final_holdout <- bt[test_index,]
dev <- bt[-test_index,]

# check if outcomes are equally distributed
prop.table(table(final_holdout$Class))
prop.table(table(dev$Class))

saveRDS(final_holdout, file="data/final_holdout.rds")
saveRDS(dev, file="data/dev.rds")

# Create another testing set for testing model performance during model development
set.seed(7)
test_index <- createDataPartition(dev$Class, times=1, p=0.1, list=FALSE)
test <- dev[test_index,]
train <- dev[-test_index,]
#check if outcomes are equally distributed
prop.table(table(train$Class))
prop.table(table(test$Class))

# save and reload data sets
saveRDS(test, file="data/test.rds")
saveRDS(train, file="data/train.rds")
test <- readRDS(file="data/test.rds")
train <- readRDS(file="data/train.rds")

##### Data exploration
# example of how to visualize underlying brain scan images
img <- jpeg::readJPEG("data/brain-tumor/Brain Tumor/Image2.jpg")
str(img)
plot(as.raster(img))
scan <- recordPlot()
scan

#Let's visualize 16 random images, half tumor, half normal
set.seed(1997)
img <- c(sample(train$Image[which(train$Class==0)] , 8), sample(train$Image[which(train$Class==1)] , 8))

# Letters signify the true class, T for "tumor" or "1" and N for "normal" or "0"
  par(mar = c(2, 1, 1, 1), xpd=TRUE)
  par(mfrow = c(4,4))
  for (i in img) {
    cat(i)
    image <- jpeg::readJPEG(paste0("data/brain-tumor/Brain Tumor/",i,".jpg"))
    title <- i
    plot(as.raster(image))
    title(main=title)
    text(x=25, y=30, ifelse(train$Class[which(train$Image==i)] == 1, "T", "N"), cex=3, col= ifelse(train$Class[which(train$Image==i)] == 1, "red", "#2e8b57"))
  }
scan_grid <- recordPlot()
pdf("fig/scan_exmpls.pdf", width=8, height=8)
print(scan_grid)
dev.off()
rm(scan_grid, i, image, img, title, bt, dev, final_holdout, test_index, dl, brain_tumor)

# reshuffle columns to accurately reflect first- and second order features and sort them according to their correlations (see further below)
train <- train %>% select(Image, Class, Mean, Variance, StdDev, Energy, ASM, Entropy, Homogeneity, Contrast, Dissimilarity, Skewness, Kurtosis, Correlation)

test <- test %>% select(Image, Class, Mean, Variance, StdDev, Energy, ASM, Entropy, Homogeneity, Contrast, Dissimilarity, Skewness, Kurtosis, Correlation)

# visualize relationship between features and outcome in a fraction of the training data set to enable visual inspection

#set margins and colors
par(mar = c(2, 2, 4, 2), xpd=TRUE, font.main=2, cex.main=1)
colors <- c("#2e8b57", 
            "#F8766D")
# plot the data in 10% of the train data set
set.seed(1)
train %>% select(-1) %>% slice_sample(prop=0.1) %>%
  plot(., pch=1, lwd=0.7, cex=0.75, col=colors[factor(.$Class)], main="Scatter plots between features and outcome")
# position of legend will vary depending on machine and installation
legend(0, 1.5, legend = c("no tumor", "tumor"),
       pch = 19,
       col = colors, cex=0.5)

all_vs_all <- recordPlot()
pdf("fig/scatter_prdc_outc.pdf", width=8, height=8)
all_vs_all
dev.off()

##### Selection of a subset of predictors using a filter method. This helps reducing computational power by removing redundant and thus low-information-content features.

# visualize the correlation between the predictors
par(mar = c(6, 6, 4, 2), xpd=TRUE, font.main=2, cex.main=1)
corel <- cor(train[,2:14])
# correlation of predictors to outcome (Class)
corel[,1]
# heatmap the correlation matrix and add matching correlation coefficients inside the cells of the matrix with a for loop

pal <-  colorRampPalette(c("blue", "white", "red"))(100)
image(corel, axes=FALSE, col = pal)
names <- names(train[,2:14])
axis(side = 1, at = seq(0,1.0, length.out=13), labels = names, las=2)
axis(side = 2, at = seq(0,1.0, length.out=13), labels = names, las=2)
text(0.5,1.1,"Correlation between features and outcome")
# for loop to add corresponding values inside the heatmap
for(i in 1:nrow(corel)) {
  for(j in 1:ncol(corel)) {
    text(((j-1)*(1/(ncol(corel)-1))), ((i-1)*(1/(nrow(corel)-1))), round(corel[i,j], 2), col="black")
  }
}
cor_plot <- recordPlot()
pdf("fig/correl_prdc_outc.pdf", width=8, height=8)
print(cor_plot)
dev.off()

# We see four correlation clusters, with correlation values > 0.70.
# We will keep only one-two features per correlation cluster and select them based on their correlation or anti-correlation to the outcome, and on their value spread, as also seen in boxplot "bp" and the "all_vs_all" scatter plot created above 
bp
dev.off()
par(mar = c(2, 2, 4, 2), xpd=TRUE, font.main=2, cex.main=1)
all_vs_all
dev.off()
cor_plot
# Mean, Variance and StdDev are highly correlated between each other. Variance has highest positive correlation to Class, but Mean is anti-correlated, we therefore keep Mean and Variance.

#Entropy, Energy, ASM and Homogeneity are highly correlated. Energy has the highest anticorrelation to class. We keep Homogeneity as well since it behaves differently from the other three predictors. We saw that when we inspected it in the feature vs. features (all_vs_all) plot. Homogeneity is scattered as opposed to a discrete curved line as seen for plots between the other three features.

# Dissimilarity and Contrast are highly correlated. Dissimilarity has higher correlation to Class and is retained.

# Kurtosis and Skewness are highly correlated. Skewness has higher correlation to Class and is retained.

# Therefore, we keep following 7 of the 12 features for training:
# Variance, Mean, Energy, Homogeneity, Skewness, Dissimilarity and Correlation.


##### We are applying the filter method for the choice of features to train a prediction model. We are selecting the features obtained as described above using correlation coefficients as the main guideline:
# Variance, Mean, Energy, Homogeneity, Skewness, Dissimilarity and Correlation.


# random prediction with Monte-Carlo simulation to confirm that the accuracy of a prediction that does not do better than chance is 50 %, since there are only two outcomes
set.seed(1997)
guess <- replicate(100, {
  random <- sample(factor(c(0, 1)), length(test$Class), replace=TRUE)
          mean(random == factor(test$Class))}
          )
mean(guess)

# We train a k-nearest neighbor model
train_knn <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data=train, method="knn", tuneGrid = data.frame(k = seq(1,301,30)))
train_knn$results
plot(train_knn)
test_knn<- predict(train_knn, newdata=test)
mean(test_knn == factor(test$Class))
#knn performs rather poorly with the highest accuracy at k = 1, suggesting overfitting.

# examine how just two features with highest correlation can separate the outcome
train %>% ggplot(aes(Energy, Homogeneity, color = factor(Class))) +
  geom_point()

#  We train a generalized linear model
train_glm <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="glm", trControl=trainControl(method="cv", number=10))
train_glm$results
train_glm$finalModel
tidy(train_glm$finalModel)
test_glm<- predict(train_glm, newdata=test)
mean(test_glm == factor(test$Class))
# Accuracy: [1] 0.9852507
miss <- which(test_glm != factor(test$Class))
miss # which scans (row indices) were misclassified

# highlight the misclassified cases
test %>% mutate(acc = ifelse(test_glm != factor(test$Class), 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class), color=factor(acc), size=factor(acc))) +
  geom_point(alpha=0.5)

# visualizing the misclassified images. We construct a function that takes the vector with the row indices of missed calls and the data set as input, and the loops through the row indices to fetch the corresponding brain scan images.
# Letters signify the true class, T for "tumor" or "1" and N for "normal" or "0"
plot_misses <- function(x, y) {
par(mar = c(2, 1, 1, 1), xpd=TRUE)
par(mfrow = c(3,3))
for (i in miss) {
  cat(paste0("Row index ", i, ", ", (y$Image[i]), " "))
  img_miss <- jpeg::readJPEG(paste0("data/brain-tumor/Brain Tumor/",y$Image[i],".jpg"))
  title <-y$Image[i]
plot(as.raster(img_miss))
title(main=title)
text(x=25, y=30, ifelse(y$Class[i] == 1, "T", "N"), cex=3, 
     col= ifelse(y$Class[i] == 1, "red", "#2e8b57"))
# assign(paste0("img_miss_", i), recordPlot())
}}
plot_misses(miss, test)

set.seed(1)
hits <- sample( which(test_glm == factor(test$Class)), 9)

# visualizing randomly selected, correctly classified images. Again, a function taking the hits vector and the data set as input.
# Letters signify the true class, T for "tumor" or "1" and N for "normal" or "0"

plot_hits <- function(x, y) {
  par(mar = c(2, 1, 1, 1), xpd=TRUE)
  par(mfrow = c(3,3))
  for (i in hits) {
    cat(paste0("Row index ", i, ", ", (y$Image[i]), " "))
    img_miss <- jpeg::readJPEG(paste0("data/brain-tumor/Brain Tumor/",y$Image[i],".jpg"))
    title <-y$Image[i]
    plot(as.raster(img_miss))
    title(main=title)
  text(x=25, y=30, ifelse(y$Class[i] == 1, "T", "N"), cex=3, 
       col= ifelse(y$Class[i] == 1, "red", "#2e8b57"))
  #assign(paste0("img_hit_", i), recordPlot())
}}
plot_hits(hits, test)

#  We train a loess model...
modelLookup("gamLoess")
# ...and optimize span
grid <- expand.grid(span = seq(0.01, 0.21, 0.025), degree=1)
# computationally intensive! optinally pre-load the model with code 12 lines below
set.seed(1)
train_gamloess<- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="gamLoess", trControl=trainControl(method="cv", number=10), tuneGrid=grid)
train_gamloess$bestTune
train_gamloess$results
plot(train_gamloess)
# span of 0.16 is best. Training is computationally intensive!
tidy(train_gamloess$finalModel)
test_gamloess<- predict(train_gamloess, newdata=test)
mean(test_gamloess == factor(test$Class))
# Accuracy: [1] 0.9911504
saveRDS(train_gamloess, "models/train_gamloess.rds")
train_gamloess<- readRDS("models/train_gamloess.rds")
test_gamloess<- predict(train_gamloess2, newdata=test)
mean(test_gamloess == factor(test$Class))

test %>% mutate(acc = ifelse(test_gamloess != factor(test$Class), 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class), color=factor(acc), size=factor(acc))) +
  geom_point(alpha=0.5)

miss <- which(test_gamloess != factor(test$Class))
set.seed(1)
hits <- sample( which(test_gamloess == factor(test$Class)), 9)
plot_misses(miss, test)
plot_hits(hits, test)
dev.off()


# We train a decision tree
set.seed(1999)
train_rpart <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="rpart", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(cp=seq(0.0,0.01, length=11)))
train_rpart$bestTune
plot(train_rpart$finalModel, margin=0.1)
text(train_rpart$finalModel, cex = 0.75)

test_rpart<- predict(train_rpart, newdata=test, type="raw")
mean(test_rpart == factor(test$Class))
# Accuracy: [1] 0.9852507
miss <- which(test_rpart != factor(test$Class))
set.seed(1)
hits <- sample( which(test_rpart == factor(test$Class)), 9)
plot_misses(miss, test)
plot_hits(hits, test)
dev.off()


# can we optimize the partitioning parameter minsplit?
part <- seq(10,100,5)
set.seed(1999)
acc <- sapply(part, function(k) { 
train_rpart <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="rpart", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(cp=seq(0.0,0.01, length=11)), minsplit=k, minbucket=5)
data.frame(part=k, cp = as.numeric(train_rpart$bestTune), accuracy =  train_rpart$results$Accuracy[which(train_rpart$results$cp ==as.numeric(train_rpart$bestTune))])
} )

# get index of maximum accuracy entry
which.max(acc[3,])
# which minsplit value corresponds to that - 50
acc[1,][which.max(acc[3,])]
# which cp value corresponds to that - 0.001
acc[2,][which.max(acc[3,])]

#trying to optimize minbucket value now
min <- seq(1,21,2)
# taking filter method for feature choice
set.seed(1999)
acc_minnode <- sapply(min, function(k) { 
  train_rpart <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="rpart", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(cp=seq(0.0,0.01, length=11)), minsplit=50, minbucket=k)
  data.frame(minnode=k, cp = as.numeric(train_rpart$bestTune), accuracy =  train_rpart$results$Accuracy[which(train_rpart$results$cp ==as.numeric(train_rpart$bestTune))])
} )

# get index of maximum accuracy entry
which.max(acc_minnode[3,])
# which minbucket value corresponds to that - 21
acc_minnode[1,][which.max(acc_minnode[1,])]
# which cp value corresponds to that - 0.001
acc_minnode[2,][which.max(acc_minnode[1,])]

#cp = 0.001 best tune, minsplit - 50, minbucket - 21
train_rpart <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="rpart", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(cp=0.001), minsplit=50, minbucket=21)

plot(train_rpart$finalModel, margin=0.1)
text(train_rpart$finalModel, cex = 0.75)
test_rpart<- predict(train_rpart, newdata=test, type="raw")
mean(test_rpart == factor(test$Class))
# Accuracy: [1] 0.9882006
miss <- which(test_rpart != factor(test$Class))
set.seed(1)
hits <- sample( which(test_rpart == factor(test$Class)), 9)
plot_misses(miss, test)
plot_hits(hits, test)
dev.off()
# optimization did not improve the prediction by much!

# We train random forest which should improve the decison tree prediction, but computationally somewhat intensive
modelLookup("rf")

set.seed(2000)
train_rf <- train(factor(Class) ~Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="rf", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(mtry=seq(1,7,1)))

plot(train_rf)
train_rf$results
train_rf$bestTune

# 4 mtry is best

#find if there is a better node size. Computationally intensive!
set.seed(2000)
nodesize <- seq(1, 51, 5)
acc <- sapply(nodesize, function(ns){
  train_rf <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="rf", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(mtry=4), nodesize=ns)$results$Accuracy
})
plot(nodesize,acc)
nodesize[which.max(acc)]
# optimal nodesize is 6

# train model with optimized parameters
set.seed(2000)
train_rf <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="rf", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(mtry=4), nodesize=6)

test_rf<- predict(train_rf, newdata=test, type="raw")
mean(test_rf == factor(test$Class))
# Accuracy: [1]  0.9911504
miss <- which(test_rf != factor(test$Class))
set.seed(1)
hits <- sample( which(test_rpart == factor(test$Class)), 9)
plot_misses(mis, test)
plot_hits(hits, test)
dev.off()

test %>% mutate(acc = ifelse(test_rf != factor(test$Class), 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class), color=factor(acc), size=factor(acc))) +
  geom_point(alpha=0.5)

# we train a gradient boosting machine model
train_gbm <-  train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="gbm")
modelLookup("gbm")
train_gbm$results
train_gbm$bestTune
#n.trees interaction.depth shrinkage n.minobsinnode
# 150                 3       0.1             10
plot(train_gbm)

train_gbm <-  train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="gbm", tuneGrid=data.frame(n.trees = 150, interaction.depth=3, shrinkage=0.1, n.minobsinnode=10) )


test_gbm<- predict(train_gbm, newdata=test, type="raw")
mean(test_gbm == factor(test$Class))
# Accuracy: [1]  0.9911504
miss <- which(test_gbm != factor(test$Class))
set.seed(1)
hits <- sample( which(test_rpart == factor(test$Class)), 9)
plot_misses(miss, test)
plot_hits(hits, test)
dev.off()


# ensembl method. convert the factors to outcome classes taking values 0 and 1.
test_glm_n <- as.numeric(test_glm)-1
test_gamloess_n <- as.numeric(test_gamloess)-1
test_rpart_n <- as.numeric(test_rpart)-1
test_rf_n <- as.numeric(test_rf)-1
test_gbm_n<- as.numeric(test_gbm)-1
# create an ensembl prediction based on majority vote
ensembl <- (test_glm_n + test_gamloess_n + test_rpart_n + test_rf_n + test_gbm_n)/5
test_ensembl <- ifelse(ensembl>0.5, 1, 0)
mean(test_ensembl == test$Class)

miss <- which(test_ensembl != factor(test$Class))
set.seed(1)
hits <- sample( which(test_rpart == factor(test$Class)), 9)
plot_misses(miss, test)
plot_hits(hits, test)
dev.off()


test %>% mutate(acc = ifelse(test_ensembl!= test$Class, 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class), color=factor(acc), size=factor(acc))) +
  geom_point(alpha=0.5)


# train models with unpartitioned "dev" data set
dev <- readRDS("data/dev.rds")

dev_glm <- train(factor(Class) ~  Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = dev, method="glm", trControl=trainControl(method="cv", number=10))

dev_gamloess<- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = dev, method="gamLoess", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(span=0.16, degree=1))

dev_rpart <- train(factor(Class) ~  Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = dev, method="rpart", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(cp=0.001), minsplit=50, minbucket=21)

dev_rf <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = dev, method="rf", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(mtry=4), nodesize=6)

dev_gbm <-  train(factor(Class) ~  Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = dev, method="gbm", tuneGrid=data.frame(n.trees = 150, interaction.depth=3, shrinkage=0.1, n.minobsinnode=10) )

final_holdout <- readRDS("data/final_holdout.rds")

# predict outcomes in final_holdout set based on models trained on dev
final_holdout_glm<- predict(dev_glm, newdata=final_holdout)
final_holdout_gamloess<- predict(dev_gamloess, newdata=final_holdout)
final_holdout_rpart<- predict(dev_rpart, newdata=final_holdout)
final_holdout_rf<- predict(dev_rf, newdata=final_holdout)
final_holdout_gbm<- predict(dev_gbm, newdata=final_holdout)

# ensembl method. convert the factors to outcome classes taking values 0 and 1.
final_holdout_glm_n <- as.numeric(final_holdout_glm)-1
final_holdout_gamloess_n <- as.numeric(final_holdout_gamloess)-1
final_holdout_rpart_n <- as.numeric(final_holdout_rpart)-1
final_holdout_rf_n <- as.numeric(final_holdout_rf)-1
final_holdout_gbm_n<- as.numeric(final_holdout_gbm)-1

# create an ensembl prediction based on majority vote
ensembl <- (final_holdout_glm_n + final_holdout_gamloess_n + final_holdout_rpart_n + final_holdout_rf_n + final_holdout_gbm_n)/5
final_holdout_ensembl <- ifelse(ensembl>0.5, 1, 0)
# assess accuracy
mean(final_holdout_ensembl == final_holdout$Class)
confusionMatrix(factor(final_holdout_ensembl), factor(final_holdout$Class))
F_meas(factor(final_holdout_ensembl), factor(final_holdout$Class))
# our models have an accuracy of 0.9841 


final_holdout %>% mutate(miss = ifelse(final_holdout_ensembl!= final_holdout$Class, 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class), color=factor(miss), size=factor(miss))) +
  geom_point(alpha=0.5)

miss <- which(final_holdout_ensembl != final_holdout$Class)
set.seed(1)
hits <- sample( which(final_holdout_ensembl == final_holdout$Class), 9)
plot_misses(miss, final_holdout)
plot_hits(hits, final_holdout)
dev.off()









# wrapper method for feature selection. using random forest to find the important variables. COMPUTATION INTENSIVE.
set.seed(1)
train_rf <- caret::train(x=train[,3:14], y=factor(train$Class), method="rf", tuneGrid = data.frame(mtry = seq(2,12,2)), trControl = trainControl(method = "cv"))
train_rf$bestTune # 4 is optimal. i.e. 4 predictors minimizes RMSE            
varImp(train_rf) # Overall
#Energy        100.0000
#Entropy        88.0045
#ASM            84.8858
#Homogeneity    47.6402
#Dissimilarity   9.4572
#Kurtosis        9.1387
#Skewness        2.6455
#Variance        1.3475
#StdDev          1.2934
#Correlation     0.5906
#Mean            0.2522
#Contrast        0.0000

# According to this wrapper method, we would definitely exclude Contrast and could exclude Correlation and Mean (less than 1.0).

saveRDS(train_rf, "models/train_randomfor.rds")
train_rf <- readRDS("models/train_randomfor.rds")

plot(train_rf)
varImp(train_rf)

predict_rf <- predict(train_rf, test, type="raw")
mean(predict_rf == factor(test$Class))
#[1] 0.9911504

confusionMatrix(factor(predict_rf), factor(test$Class))





