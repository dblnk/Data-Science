#####################################################################
### Development of a machine learning model aiming to predict the ###
### presence of a brain tumor in brain MRI scan images            ###
#####################################################################

# you can clone the project from my github repository: 
# https://github.com/dblnk/Data-Science/tree/master/project-braintumor

# Or you build it quickly yourself by setting the working directory to a directory with "data", "fig" and "model" sub-directories present, then:

# If you are registered and logged into Kaggle, then download the archived data set from this URL into your data sub-directory (approx. 15 MB):
# https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor/download?datasetVersionNumber=3

# then extract the archive with this code
dl <- "data/archive.zip"
brain_tumor <- "data/brain-tumor/Brain Tumor.csv"
if(!file.exists(brain_tumor))
  unzip(dl, exdir="data/brain-tumor")

# Alternatively, use this code to download the data from my github repository:

dl <- "data/brain-tumor.zip"
if(!file.exists(dl))
  download.file("https://github.com/dblnk/Data-Science/raw/master/project-braintumor/data/brain-tumor.zip", dl)
brain_tumor <- "data/brain-tumor/Brain Tumor.csv"
if(!file.exists(brain_tumor))
  unzip(dl, exdir="data/brain-tumor")

# the data set was derived by the kaggle submitter JAKESH BOHAJU from https://www.smir.ch/BRATS/Start2015#!#evaluation . The paper describing the BRATS brain segmentation challenge is available under https://ieeexplore.ieee.org/document/6975210/ by Menze et al., The Multimodal Brain TumorImage Segmentation Benchmark (BRATS), IEEE Trans. Med. Imaging, 2015. 

##### If downloaded dircetly from KAGGLE: then after succesfully unzipping the file, change the structure of the subdirectories in the terminal using this Unix code:
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
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(ggrepel)) install.packages("ggrepel", repos = "http://cran.us.r-project.org")

##### STEP 1: Data pre-processing #####
# Load the data and make numbers more readable
options(digits=7)

brain_tumor <- "data/brain-tumor/Brain Tumor.csv"
bt <- read.csv2(file=brain_tumor, sep=",", dec=".")

#Image column defines image name and Class column defines if the image shows a tumor or not (1 = Tumor, 0 = Non-Tumor)

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

# reshuffle columns to accurately reflect first- and second order features and sort them according to their correlations (see STEP 2 further below)
train <- train %>% select(Image, Class, Mean, Variance, StdDev, Energy, ASM, Entropy, Homogeneity, Contrast, Dissimilarity, Skewness, Kurtosis, Correlation)

test <- test %>% select(Image, Class, Mean, Variance, StdDev, Energy, ASM, Entropy, Homogeneity, Contrast, Dissimilarity, Skewness, Kurtosis, Correlation)

##### STEP 2: Data exploration #####
# example of how to visualize and save underlying brain scan images
img <- jpeg::readJPEG("data/brain-tumor/Brain Tumor/Image2.jpg")
str(img)
plot(as.raster(img))
scan <- recordPlot()
scan

#Let's visualize 16 random images, half tumor, half normal
set.seed(1997)
img <- c(sample(train$Image[which(train$Class==0)] , 8), sample(train$Image[which(train$Class==1)] , 8))

# Letters signify the true class, T for "tumor" or "1" and N for "normal" or "0"
par(mar=c(0,0,1,0), mfrow = c(4,4), xpd=TRUE)
  for (i in img) {
    cat(i)
    image <- jpeg::readJPEG(paste0("data/brain-tumor/Brain Tumor/",i,".jpg"))
    title <- i
    plot(as.raster(image))
    title(main=title)
    text(x=25, y=30, ifelse(train$Class[which(train$Image==i)] == 1, "T", "N"), cex=3, col= ifelse(train$Class[which(train$Image==i)] == 1, "red", "#2e8b57"))
  }

# save image grid
scan_grid <- recordPlot()
pdf("fig/scan_exmpls.pdf", width=8, height=8)
print(scan_grid)
dev.off()


ds_theme_set()
#Let's look at the value distributions
bp <- bt %>% pivot_longer(Mean:Correlation, names_to="parameter", values_to="value") %>% 
  ggplot(aes(parameter, value, color=factor(Class, labels=c("normal","tumor")))) + 
  geom_boxplot(outlier.size=0.2) +
  scale_color_manual(values = c("#2e8b57", "#F8766D"))+
  guides(color=guide_legend(title=NULL))+
  scale_y_continuous(trans="log10")+
  ggtitle("Distribution of features between outcomes")+
  xlab("")+
  theme(plot.title = element_text(hjust=0.5, size=12))+
  theme(axis.text.x = element_text(angle=90, hjust=1.0))
bp
ggsave("fig/boxplot_prd_outc.pdf")

rm(scan_grid, i, image, img, title, dev, bt, final_holdout, test_index, dl, brain_tumor)

# reshuffle columns to accurately reflect first- and second order features and sort them according to their correlations, if not already done
train <- train %>% select(Image, Class, Mean, Variance, StdDev, Energy, ASM, Entropy, Homogeneity, Contrast, Dissimilarity, Skewness, Kurtosis, Correlation)

test <- test %>% select(Image, Class, Mean, Variance, StdDev, Energy, ASM, Entropy, Homogeneity, Contrast, Dissimilarity, Skewness, Kurtosis, Correlation)

# visualize relationship between features and outcome in a fraction of the training data set to enable visual inspection

#set margins and colors
par(mar=c(0,0,1,0), xpd=TRUE, font.main=2, cex.main=1)
colors <- c("#2e8b57", 
            "#F8766D")
# pairwise scatter plot the data in 10% of the train data set
set.seed(1)
train %>% select(-1) %>% slice_sample(prop=0.1) %>%
  pairs(., pch=19, lwd=0.5, cex=0.6, col=colors[factor(.$Class)], lower.panel=NULL, main="Scatter plots between features and outcome")
# position of legend might vary depending on session, machine and installation, examine position of corners in "leg" and adjust the legend coordinates
leg <- legend("left", legend = c("no tumor", "tumor"),
              pch = 19,
              col = colors, plot=FALSE)
legend(x=c(0, 0.21), y=c(0.30, 0.51), legend = c("no tumor", "tumor"),
       pch = 19,
       col = colors)
# save plot
all_vs_all <- recordPlot()
pdf("fig/scatter_prdc_outc.pdf", width=8, height=8)
all_vs_all
dev.off()

##### STEP 3: Selection of a subset of predictors using a filter method. This helps reducing computational power by removing redundant and thus low-information-content features.#####

### visualize the correlation between the predictors
# create correlation matrix
corel <- cor(train[,2:14])
# correlation of predictors to outcome (Class)
corel[,1]
# heatmap the correlation matrix and add matching correlation coefficients inside the cells of the matrix with a for loop
par(mar = c(5, 5, 1.5, 2), xpd=TRUE, font.main=2, cex.main=1)
pal <-  colorRampPalette(c("blue", "white", "red"))(100)
image(corel, axes=FALSE, col = pal)
names <- names(train[,2:14])
axis(side = 1, at = seq(0,1.0, length.out=13), labels = names, las=2, cex.axis=0.75)
axis(side = 2, at = seq(0,1.0, length.out=13), labels = names, las=2, cex.axis=0.75)
title(main="Correlation between features and outcome")
# for loop to add corresponding values inside the heatmap
for(i in 1:nrow(corel)) {
  for(j in 1:ncol(corel)) {
    text(((j-1)*(1/(ncol(corel)-1))), ((i-1)*(1/(nrow(corel)-1))), round(corel[i,j], 2), col="black", cex=0.7)
  }
}
# save plot
cor_plot <- recordPlot()
pdf("fig/correl_prdc_outc.pdf", width=8, height=8)
print(cor_plot)
dev.off()

# We see four correlation clusters, with correlation values > 0.70.
# We will keep only one-two features per correlation cluster and select them based on their correlation or anti-correlation to the outcome, and on their value spread (deterministic vs stochastic), as  seen in the "all_vs_all" scatter plot created above 
par(mar=c(0,0,1,0), xpd=TRUE, font.main=2, cex.main=1)
all_vs_all
dev.off()
cor_plot
# Mean, Variance and StdDev are highly correlated between each other. Variance has highest positive correlation to Class, but Mean is anti-correlated, we therefore keep Mean and Variance.

#Entropy, Energy, ASM and Homogeneity are highly correlated. Energy has the highest anti-correlation to class. We keep Homogeneity as well since it behaves differently from the other three predictors (stochastics and not deterministic). We saw that when we inspected it in the feature vs. features (all_vs_all) plot. Homogeneity is scattered as opposed to a discrete curved line as seen for plots between the other three features.

# Dissimilarity and Contrast are highly correlated. Dissimilarity has higher correlation to Class and is retained.

# Kurtosis and Skewness are highly correlated. Skewness has higher correlation to Class and is retained.

# Therefore, we keep following 7 of the 12 features for training:
# Variance, Mean, Energy, Homogeneity, Skewness, Dissimilarity and Correlation.

# random prediction with Monte-Carlo simulation to confirm that the accuracy of a prediction that does not do better than chance is 50 %, since there are only two outcomes
set.seed(1997)
guess <- replicate(100, {
  random <- sample(factor(c(0, 1)), length(test$Class), replace=TRUE)
          mean(random == factor(test$Class))}
          )
mean(guess)

##### STEP 4: Training models #####

### STEP 4.1
# We tune a k-nearest neighbor model
set.seed(1995)
train_knn <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data=train, method="knn", trControl=trainControl(method="cv", number=10), tuneGrid = data.frame(k = seq(1,51,2)))
train_knn$results
train_knn$bestTune
plot(train_knn)

# train with k = 5
set.seed(1995)
train_knn <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data=train, method="knn", trControl=trainControl(method="cv", number=10), tuneGrid = data.frame(k = 5))
# evaluate on the test set
test_knn<- predict(train_knn, newdata=test)
acc_knn <- mean(test_knn == factor(test$Class))
miss_knn <- which(test_knn != factor(test$Class)) # which scans (row indices) were misclassified by knn

#knn performs rather poorly (compare to models below) with the highest accuracy of 83.2% at k = 5, suggesting overfitting.

# examine how two features from different correlation clusters, with highest (absolute) correlation to outcome, i.e. Energy and Dissimilarity, can separate the outcome and how the missed predictions map
plot_knn <- test %>% mutate(acc = ifelse(test_knn != factor(test$Class), 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class, labels=c("normal","tumor")), color=factor(acc, labels=c("hit","miss")), size=factor(acc))) +
  geom_point(alpha=0.5)+  
  scale_color_manual(values = c("#2e8b57", "#F8766D"))+
  scale_size_manual(values = c(2,4))+
  ggtitle("kNN")+
  theme(plot.title = element_text(size=12,hjust=0.5))+
  guides(color=guide_legend(title="prediction"), shape=guide_legend(title="True outcome"), size=FALSE)
plot_knn


### STEP 4.2
#  We train a generalized linear model
train_glm <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="glm", trControl=trainControl(method="cv", number=10))

train_glm$results
tidy(train_glm$finalModel)

#evaluate on the test set
test_glm<- predict(train_glm, newdata=test)
acc_glm <- mean(test_glm == factor(test$Class))
acc_glm
# Accuracy: [1] 0.9852507
miss_glm <- which(test_glm != factor(test$Class)) # which scans (row indices) were misclassified by glm
miss_glm 

# highlight the misclassified cases
plot_glm <- test %>% mutate(acc = ifelse(test_glm != factor(test$Class), 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class, labels=c("normal","tumor")), color=factor(acc, labels=c("hit","miss")), size=factor(acc))) +
  geom_point(alpha=0.5)+  
  scale_color_manual(values = c("#2e8b57", "#F8766D"))+
  scale_size_manual(values = c(2,4))+
  ggtitle("glm")+
  theme(plot.title = element_text(size=12,hjust=0.5))+
  guides(color=guide_legend(title="prediction"), shape=guide_legend(title="True outcome"), size=FALSE)
plot_glm

grid.arrange(plot_knn, plot_glm, ncol=2)

### visualizing the misclassified MRI images. We construct a function that takes the vector with the row indices of missed calls and the data set as input, and the loops through the row indices to fetch the corresponding brain scan images.
# Letters signify the true class, T for "tumor" or "1" and N for "normal" or "0"
plot_misses <- function(x, y) {
  par(mar = c(0, 0, 1, 0), xpd=TRUE, mfrow = c(3,3))
  for (i in x) {
    cat(paste0("Row index ", i, ", ", (y$Image[i]), " "))
    img_miss <- jpeg::readJPEG(paste0("data/brain-tumor/Brain Tumor/",y$Image[i],".jpg"))
    title <-y$Image[i]
    plot(as.raster(img_miss))
    title(main=title)
    text(x=25, y=30, ifelse(y$Class[i] == 1, "T", "N"), cex=3, 
         col= ifelse(y$Class[i] == 1, "red", "#2e8b57"))
    # assign(paste0("img_miss_", i), recordPlot())
  }}

# random 9 scan that were correctly predicted by glm
set.seed(1)
hits <- sample( which(test_glm == factor(test$Class)), 9)

# visualizing randomly selected, correctly classified images. Again, a function taking the hits vector and the data set as input.
# Letters signify the true class, T for "tumor" or "1" and N for "normal" or "0"

plot_hits <- function(x, y) {
  par(mar = c(0, 0, 1, 0), xpd=TRUE, mfrow = c(3,3))
  for (i in x) {
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

### STEP 4.3
#  We train a loess model...
modelLookup("gamLoess")
# ...and optimize span
grid <- expand.grid(span = seq(0.01, 0.21, 0.025), degree=1)
# computationally intensive! optionally pre-load the model with the code provided 18 lines below
set.seed(1)
train_gamloess<- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="gamLoess", trControl=trainControl(method="cv", number=10), tuneGrid=grid)
train_gamloess$bestTune
train_gamloess$results
plot(train_gamloess)
tidy(train_gamloess$finalModel)

# span of 0.16 is best. Training is computationally intensive! Better pre-load the model.
# code for training with span of 0.16
set.seed(1)
train_gamloess<- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="gamLoess", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(span = 0.16, degree=1))

#evaluate on test set
test_loess<- predict(train_gamloess, newdata=test)
 mean(test_loess == factor(test$Class))
# Accuracy: [1] 0.9911504

saveRDS(train_gamloess, "models/train_gamloess.rds")

train_gamloess<- readRDS("models/train_gamloess.rds")
#evaluate loess model on test set
test_loess<- predict(train_gamloess, newdata=test)
acc_loess <- mean(test_loess == factor(test$Class))
acc_loess

#misclassified images
miss_loess <- which(test_loess != factor(test$Class))
set.seed(1)
hits <- sample( which(test_loess == factor(test$Class)), 9)
plot_misses(miss_loess, test)
plot_hits(hits, test)
dev.off()

#plot failed predicitons
plot_loess <- test %>% mutate(acc = ifelse(test_loess != factor(test$Class), 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class, labels=c("normal","tumor")), color=factor(acc, labels=c("hit","miss")), size=factor(acc))) +
  geom_point(alpha=0.5)+  
  scale_color_manual(values = c("#2e8b57", "#F8766D"))+
  scale_size_manual(values = c(2,4))+
  ggtitle("loess")+
  theme(plot.title = element_text(size=12,hjust=0.5))+
  guides(color=guide_legend(title="prediction"), shape=guide_legend(title="True outcome"), size=FALSE)
plot_loess

### STEP 4.4
# We tune a decision tree
set.seed(1999)
train_rpart <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="rpart", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(cp=seq(0.0,0.01, length=11)))
train_rpart$bestTune
plot(train_rpart$finalModel, margin=0.1)
text(train_rpart$finalModel, cex = 0.75)

test_rpart<- predict(train_rpart, newdata=test, type="raw")
mean(test_rpart == factor(test$Class))
# Accuracy: [1] 0.9852507

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

# Train a deicision tree with optimized parameters cp = 0.001 best tune, minsplit - 50, minbucket - 21
train_rpart <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="rpart", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(cp=0.001), minsplit=50, minbucket=21)

# plot the decision tree
par(mar = c(0, 0, 1, 0), xpd=TRUE, font.main=2, cex.main=0.85)
plot(train_rpart$finalModel, margin=0.02)
text(train_rpart$finalModel, cex = 0.8)
title(main="Decision tree")
tree <- recordPlot()
pdf("fig/train_tree.pdf", width=8, height=8)
tree
dev.off()

#evalaute on the test set
test_rpart<- predict(train_rpart, newdata=test, type="raw")
acc_rpart <- mean(test_rpart == factor(test$Class))
acc_rpart
# Accuracy: [1] 0.9882006
# optimization did not improve the prediction by much!

# misclassified images
miss_rpart <- which(test_rpart != factor(test$Class))
set.seed(1)
hits <- sample( which(test_rpart == factor(test$Class)), 9)
plot_misses(miss_rpart, test)
plot_hits(hits, test)
dev.off()

plot_rpart <- test %>% mutate(acc = ifelse(test_rpart != factor(test$Class), 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class, labels=c("normal","tumor")), color=factor(acc, labels=c("hit","miss")), size=factor(acc))) +
  geom_point(alpha=0.5)+  
  scale_color_manual(values = c("#2e8b57", "#F8766D"))+
  scale_size_manual(values = c(2,4))+
  ggtitle("Class. tree")+
  theme(plot.title = element_text(size=12,hjust=0.5))+
  guides(color=guide_legend(title="prediction"), shape=guide_legend(title="True outcome"), size=FALSE)
plot_rpart
  
  
### STEP 4.5
# We tune a random forest which should improve the decision tree validity, but computationally somewhat intensive
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

# Train the model with optimized parameters
set.seed(2000)
train_rf <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="rf", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(mtry=4), nodesize=6)
varImp(train_rf)

#evaluate on the test set 
test_rf<- predict(train_rf, newdata=test, type="raw")
acc_rf <- mean(test_rf == factor(test$Class))
acc_rf
# Accuracy: [1]  0.9911504

miss_rf <- which(test_rf != factor(test$Class))
set.seed(1)
hits <- sample( which(test_rf == factor(test$Class)), 9)
plot_misses(miss_rf, test)
plot_hits(hits, test)
dev.off()

plot_rf <- test %>% mutate(acc = ifelse(test_rf != factor(test$Class), 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class, labels=c("normal","tumor")), color=factor(acc, labels=c("hit","miss")), size=factor(acc))) +
  geom_point(alpha=0.5)+  
  scale_color_manual(values = c("#2e8b57", "#F8766D"))+
  scale_size_manual(values = c(2,4))+
  ggtitle("Random forest")+
  theme(plot.title = element_text(size=12,hjust=0.5))+
  guides(color=guide_legend(title="prediction"), shape=guide_legend(title="True outcome"), size=FALSE)
plot_rf

### STEP 4.6
# we tune a gradient boosting machine model
train_gbm <-  train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="gbm", trControl=trainControl(method="cv", number=10), distribution="bernoulli" )
modelLookup("gbm")
summary(train_gbm)
train_gbm$results
train_gbm$bestTune
#n.trees interaction.depth shrinkage n.minobsinnode
# 150                 3       0.1             10
plot(train_gbm)

# We train a GBM model with optimal tuning parameters
train_gbm <-  train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = train, method="gbm", tuneGrid=data.frame(n.trees = 150, interaction.depth=3, shrinkage=0.1, n.minobsinnode=10), trControl=trainControl(method="cv", number=10), distribution="bernoulli" )
dev.off()
summary(train_gbm)

#evaluate on the test set
test_gbm <- predict(train_gbm, newdata=test, type="raw")
acc_gbm <- mean(test_gbm == factor(test$Class))
acc_gbm
# Accuracy: [1]  0.9911504

# get misclassified images
miss_gbm <- which(test_gbm != factor(test$Class))
set.seed(1)
hits <- sample( which(test_gbm == factor(test$Class)), 9)
plot_misses(miss_gbm, test)
plot_hits(hits, test)
dev.off()

plot_gbm <- test %>% mutate(acc = ifelse(test_gbm != factor(test$Class), 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class, labels=c("normal","tumor")), color=factor(acc, labels=c("hit","miss")), size=factor(acc))) +
  geom_point(alpha=0.5)+  
  scale_color_manual(values = c("#2e8b57", "#F8766D"))+
  scale_size_manual(values = c(2,4))+
  ggtitle("Gradient boosting machine")+
  theme(plot.title = element_text(size=12,hjust=0.5))+
  guides(color=guide_legend(title="prediction"), shape=guide_legend(title="True outcome"), size=FALSE)
plot_gbm

# plot all 6 model predictions
grid.arrange(plot_knn, plot_glm, plot_loess, plot_rpart, plot_rf, plot_gbm, nrow = 2, ncol =3)
# or
test %>% select(Class, Energy, Dissimilarity) %>% 
  mutate(kNN = test_knn, GLM = test_glm, Loess = test_loess, Class.Tree = test_rpart, Random.Forest = test_rf, Grad.Boost.Machine = test_gbm) %>% 
  pivot_longer(-c(Class, Energy, Dissimilarity), values_to="Prediction", names_to="ML_model") %>%
  mutate(acc = ifelse(Prediction!= factor(Class), 1, 0), ML_model = factor(ML_model, levels=c("kNN", "GLM", "Loess", "Class.Tree", "Random.Forest", "Grad.Boost.Machine"))) %>% 
  ggplot(aes(Energy, Dissimilarity, shape = factor(Class, labels=c("normal","tumor")), color=factor(acc, labels=c("hit","miss")), size=factor(acc))) +
  geom_point(alpha=0.5)+  
  scale_color_manual(values = c("#2e8b57", "red"))+
  scale_size_manual(values = c(1,3))+
  facet_wrap("ML_model", nrow=2, ncol=3)+
  guides(color=guide_legend(title="prediction"), shape=guide_legend(title="True outcome"), size=FALSE)
ggsave("fig/train_models_DivsEn.pdf", width=8, height=6)

#compare accuracies
acc_compare <- data.frame(Model = c("kNN", "GLM", "Loess", "Class. tree", "Random forest", "GBM", "Ensemble"), Accuracy = c(acc_knn, acc_glm, acc_loess, acc_rpart, acc_rf, acc_gbm, acc_ensemble))
acc_compare 

# which and what proportion of misclassified cases from the ensemble-integrated models are correctly predicted by kNN
knn_vs_other <- 1 - mean(union(union(union(union(miss_glm, miss_loess), miss_rpart),miss_rf),miss_gbm)  %in%  miss_knn)
knn_vs_other

##### STEP 5: ensemble vote of models #####
# ensemble method. Exluding knn-model. convert the factors to outcome classes taking values 0 and 1.
test_glm_n <- as.numeric(test_glm)-1
test_loess_n <- as.numeric(test_loess)-1
test_rpart_n <- as.numeric(test_rpart)-1
test_rf_n <- as.numeric(test_rf)-1
test_gbm_n<- as.numeric(test_gbm)-1
# create an ensemble prediction based on majority vote
ensemble <- (test_glm_n + test_loess_n + test_rpart_n + test_rf_n + test_gbm_n)/5
test_ensemble <- ifelse(ensemble>0.5, 1, 0)
acc_ensemble <- mean(test_ensemble == test$Class)

miss_ens <- which(test_ensemble != factor(test$Class))
set.seed(1)
hits <- sample( which(test_rpart == factor(test$Class)), 9)
plot_misses(miss_ens, test)
plot_hits(hits, test)
dev.off()

# plot failrd predicitons of ensemble (same as rf and gbm)
test %>% mutate(acc = ifelse(test_ensemble!= test$Class, 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class, labels=c("normal","tumor")), color=factor(acc, labels=c("hit","miss")), size=factor(acc))) +
  geom_point(alpha=0.5)+  
  scale_color_manual(values = c("#2e8b57", "#F8766D"))+
  scale_size_manual(values = c(2,4))+
  ggtitle("Ensemble")+
  theme(plot.title = element_text(size=12,hjust=0.5))+
  guides(color=guide_legend(title="prediction"), shape=guide_legend(title="True outcome"), size=FALSE)

#clean up
rm(leg, plot_gbm, plot_glm, plot_knn, plot_loess, plot_rf, plot_rpart, train, train_gamloess, train_gbm, train_glm, train_knn, train_rf, train_rf, test_ensemble, test_gbm, test_glm, test_knn, test_loess, test_rf, test_rpart, test_gbm_n,test_glm_n, test_loess_n, test_rf_n, test_rpart_n)

##### STEP 6: Train final model #####
# train models with unpartitioned "dev" data set
dev <- readRDS("data/dev.rds")

set.seed(1995)
dev_knn <- train(factor(Class) ~  Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = dev, method="knn", tuneGrid=data.frame(k=5), trControl=trainControl(method="cv", number=10))

set.seed(1995)
dev_glm <- train(factor(Class) ~  Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = dev, method="glm", trControl=trainControl(method="cv", number=10))

# computationally intensive, alternative load model with code below
set.seed(1995)
dev_gamloess<- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = dev, method="gamLoess", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(span=0.16, degree=1))
saveRDS(dev_gamloess, "models/dev_gamloess.rds")
dev_gamloess <- readRDS("models/dev_gamloess.rds")

set.seed(1995)
dev_rpart <- train(factor(Class) ~  Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = dev, method="rpart", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(cp=0.001), minsplit=50, minbucket=21)

set.seed(1995)
dev_rf <- train(factor(Class) ~ Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = dev, method="rf", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(mtry=4), nodesize=6)
varImp(dev_rf)

set.seed(1995)
dev_gbm <-  train(factor(Class) ~  Variance + Mean + Energy + Homogeneity + Skewness + Dissimilarity + Correlation, data = dev, method="gbm", trControl=trainControl(method="cv", number=10), tuneGrid=data.frame(n.trees = 150, interaction.depth=3, shrinkage=0.1, n.minobsinnode=10) )

##### STEP 7: Evalaute model performance #####
final_holdout <- readRDS("data/final_holdout.rds")

# predict outcomes in final_holdout set based on models trained on dev
final_knn <- predict(dev_knn, newdata=final_holdout)
acc_final_knn <- mean(final_knn == factor(final_holdout$Class))
final_miss_knn <- which(final_knn != factor(final_holdout$Class))

final_glm <- predict(dev_glm, newdata=final_holdout)
acc_final_glm <- mean(final_glm == factor(final_holdout$Class))
final_miss_glm <- which(final_glm != factor(final_holdout$Class))

final_loess<- predict(dev_gamloess, newdata=final_holdout)
acc_final_loess<- mean(final_loess == factor(final_holdout$Class))
final_miss_loess <- which(final_loess != factor(final_holdout$Class))

final_rpart<- predict(dev_rpart, newdata=final_holdout)
acc_final_rpart <- mean(final_rpart == factor(final_holdout$Class))
final_miss_rpart <- which(final_rpart != factor(final_holdout$Class))

final_rf<- predict(dev_rf, newdata=final_holdout)
acc_final_rf <- mean(final_rf == factor(final_holdout$Class))
final_miss_rf <- which(final_rf != factor(final_holdout$Class))

final_gbm<- predict(dev_gbm, newdata=final_holdout)
acc_final_gbm<- mean(final_gbm == factor(final_holdout$Class))
final_miss_gbm <- which(final_gbm != factor(final_holdout$Class))

# visualize the correct and missed predicitons in each model
final_holdout %>% select(Class, Energy, Dissimilarity) %>% 
  mutate(kNN = final_knn, GLM = final_glm, Loess = final_loess, Class.Tree = final_rpart, Random.Forest = final_rf, Grad.Boost.Machine = final_gbm) %>% 
  pivot_longer(-c(Class, Energy, Dissimilarity), values_to="Prediction", names_to="ML_model") %>%
  mutate(acc = ifelse(Prediction!= factor(Class), 1, 0), ML_model = factor(ML_model, levels=c("kNN", "GLM", "Loess", "Class.Tree", "Random.Forest", "Grad.Boost.Machine"))) %>% 
  ggplot(aes(Energy, Dissimilarity, shape = factor(Class, labels=c("normal","tumor")), color=factor(acc, labels=c("hit","miss")), size=factor(acc))) +
  geom_point(alpha=0.5)+
  scale_y_continuous(trans="log2")+
  scale_color_manual(values = c("#2e8b57", "red"))+
  scale_size_manual(values = c(2,4))+
  facet_wrap("ML_model", nrow=2, ncol=3)+
  guides(color=guide_legend(title="prediction"), shape=guide_legend(title="True outcome"), size=FALSE)
ggsave("fig/final_models_DivsEn.pdf", width=8, height=6)

# convert the factors to outcome classes taking values 0 and 1.
final_glm_n <- as.numeric(final_glm)-1
final_loess_n <- as.numeric(final_loess)-1
final_rpart_n <- as.numeric(final_rpart)-1
final_rf_n <- as.numeric(final_rf)-1
final_gbm_n<- as.numeric(final_gbm)-1

# create an ensemble prediction based on majority vote
ensemble <- (final_glm_n + final_loess_n + final_rpart_n + final_rf_n + final_gbm_n)/5
final_ensemble <- ifelse(ensemble>0.5, 1, 0)
# assess accuracy
acc_final_ensemble <- mean(final_ensemble == final_holdout$Class)
confusionMatrix(factor(final_ensemble), factor(final_holdout$Class))
F_meas(factor(final_ensemble), factor(final_holdout$Class))
# our ensemble model has an accuracy of 0.9841 

final_miss_ens <- which(final_ensemble != factor(final_holdout$Class)) # which scans (row indices) were misclassified by ensemble
set.seed(1)
final_hits_ens <- sample(which(final_ensemble == factor(final_holdout$Class)), 9) # which scans were correctly classified


# visualizing ensemble performance
final_holdout %>% mutate(acc = ifelse(final_ensemble!= final_holdout$Class, 1, 0)) %>% ggplot(aes(Energy, Dissimilarity, shape = factor(Class, labels=c("normal","tumor")), color=factor(acc, labels=c("hit","miss")), size=factor(acc))) +
  geom_point(alpha=0.5)+  
  scale_color_manual(values = c("#2e8b57", "red"))+
  scale_size_manual(values = c(2,4))+
  ggtitle("Ensemble in validation set")+
  scale_y_continuous(trans="log2")+
  guides(color=guide_legend(title="prediction"), shape=guide_legend(title="True outcome"), size=FALSE)+
  theme(plot.title = element_text(hjust=0.5, size=12))
ggsave("fig/final_ensemble_DivsEn.pdf", width=8, height=6)

#compare accuracies
acc_final_compare <- data.frame(Model = c("kNN", "GLM", "Loess", "Class. tree", "Random forest", "GBM", "Ensemble"), Accuracy = c(acc_final_knn, acc_final_glm, acc_final_loess, acc_final_rpart, acc_final_rf, acc_final_gbm, acc_final_ensemble))
acc_final_compare


# which and what proportion of misclassified cases from the ensemble-integrated models are correctly predicted by kNN
final_knn_vs_other <- 1 -  mean(union(union(union(union(final_miss_glm, final_miss_loess), final_miss_rpart),final_miss_rf),final_miss_gbm) %in% final_miss_knn)
final_knn_vs_other


##### STEP 8: MRI SCAN EVALUATION #####

# visualizing combination of all MRI scans that were misclassified by the ensemble-model during training plus during validation. 
plot_misses(final_miss_ens, final_holdout)
# add the three misclassified images from training ensemble to the remaining par(mrow) slots with this function
plot_misses_compl <- function(x, y) {
  for (i in x) {
    # cat(paste0("Row index ", i, ", ", (y$Image[i]), " "))
    img_miss <- jpeg::readJPEG(paste0("data/brain-tumor/Brain Tumor/",y$Image[i],".jpg"))
    title <-y$Image[i]
    plot(as.raster(img_miss))
    title(main=title)
    text(x=25, y=30, ifelse(y$Class[i] == 1, "T", "N"), cex=3, 
         col= ifelse(y$Class[i] == 1, "red", "#2e8b57"))
    # assign(paste0("img_miss_", i), recordPlot())
  }}

plot_misses_compl(miss_ens, test)
plots_miss_ens <- recordPlot()
pdf("fig/ensemble_fail.pdf", width=8, height=8)
plots_miss_ens
dev.off()

# visualize correctly classified images
plot_hits(final_hits_ens, final_holdout)
plots_hits_ens <- recordPlot()
pdf("fig/ensemble_success.pdf", width=8, height=8)
plots_hits_ens
dev.off()

# plot a presumable sequence of brain slices from same individual that includes two of the failed predictions
brain_tumor <- "data/brain-tumor/Brain Tumor.csv"
bt <- read.csv2(file=brain_tumor, sep=",", dec=".")
img <- paste0("Image",3056:3064)

# Letters signify the true class, T for "tumor" or "1" and N for "normal" or "0"
par(mar=c(0,0,1,0), mfrow = c(3,4), xpd=TRUE)
for (i in img) {
  image <- jpeg::readJPEG(paste0("data/brain-tumor/Brain Tumor/",i,".jpg"))
  title <- i
  plot(as.raster(image))
  title(main=title)
  text(x=25, y=30, ifelse(bt$Class[which(bt$Image==i)] == 1, "T", "N"), cex=3, col= ifelse(bt$Class[which(bt$Image==i)] == 1, "red", "#2e8b57"))
}
stack <- recordPlot()
pdf("fig/slice_stack.pdf", width=8, height=8)
stack
dev.off()


