##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "data/ml-10M.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
# If acceesing the URL does not work, please download the file manually by coping the URL and pasting in your web browser. Then unpack it into your working directory into "data" folder.

ratings_file <- "data/ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "data/ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed, ratings_file, movies_file)


setwd("/Data-Science/project-movielens")

#We save the data sets in order to load them directly in the future, when restarting the session.

saveRDS(edx, "data/edx.rds")
saveRDS(final_holdout_test, "data/final_holdout_test.rds")
rm(edx, final_holdout_test)

edx <- readRDS("data/edx.rds")

if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(purrr)) install.packages("purrr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")


# First we split the edx data into train and test set.
set.seed(1998)
test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
train <- edx[-test_index, ]
temp <- edx[test_index, ]

# verify if all movieId and userId values are present in both train and test sets.

test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Since there are some removed row. Add the removed rows from test set back into train set.
removed <- anti_join(temp, test)
train <- rbind(train, removed)

rm(test_index, temp, removed)

options(digits=7)

# We will undertake an agnostic approach, i.e. we have no idea what could influence the rating of a movie and we even don't know what a rating of, e.g., 1 versus a rating of 5 really means, i.e. which movies we want to recommend to a viewer or subscriber of the streaming or video rental service. 

# Let's start to investigate visually if any particular feature is correlated with rating

### EFFECT OF THE GENRE ###

# We want to extract the single genres that the movies are associated with. Eventually, we want to look at those movies that have only one genre assigned, to disentangle the genre ambiguities and explore if singular genres have an influence on the rating outcome or if rating does not depend on the genre.

# Let's look at how the ratings are distributed across the unambiguous genre categories. We will average the ratings that each movie has received. And we will order the factor levels of genre by the ascending order of their median ratings.

avg_genre_rating <- train %>% filter(str_detect(genres, "^\\w+\\-?\\w+\\s*$")) %>%
  group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), genre = genres[1]) %>% group_by(genre) %>%
  summarize(avg_genre_rating = median(avg_rating), n = n()) %>% mutate(genre = reorder(genre, avg_genre_rating)) 

ds_theme_set()

train %>% filter(str_detect(genres, "^\\w+\\-?\\w+\\s*$")) %>%
  group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), genre = genres[1]) %>%
  mutate(genre = factor(genre, levels=levels(avg_genre_rating$genre))) %>%
  ggplot(aes(genre, avg_rating)) + 
  geom_boxplot(outlier.shape = NA) + 
  geom_jitter(alpha=0.2, size=1)+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  ylab("Average movie rating")

ggsave("figs/genre-singl_vs_rating.pdf")

# It is apparent from this plot that some genres, such as "Children" or "Fantasy" are filled with only a few movies, hence they are likely to be rarely the only genre attributes to a movie. Therefore, we will not be able to reliably estimate the effect of a singular genre for most genres. However, to make a general case we will have a look if there are any significant differences between categories that are sufficiently crowded (n>50), i.e. "Horror","Sci-Fi", "Action", Comedy", "Thriller", "Western", "Drama" and "Documentary".

genre_crowded <- train %>% filter(str_detect(genres, "^\\w+\\-?\\w+\\s*$")) %>%
  group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), genre = genres[1]) %>%
  mutate(genre = factor(genre, levels=levels(avg_genre_rating$genre))) %>%
  group_by(genre) %>% tally() %>% arrange(desc(n)) %>% filter(n>=50) %>%pull(genre)

stats <- train %>% filter(str_detect(genres, "^\\w+\\-?\\w+\\s*$")) %>%
  group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), genre = genres[1]) %>%
  mutate(genre = factor(genre, levels=levels(avg_genre_rating$genre))) %>% filter(genre %in% genre_crowded)


# Let's plot the selected genres.

stats %>%
  ggplot(aes(genre, avg_rating)) + 
  geom_boxplot(outlier.shape = NA) + 
  geom_jitter(alpha=0.2, size=1)+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  ylab("Average movie rating")

ggsave("figs/genre-singl-crwd_vs_rating.pdf")
#First we check for normality
stats %>%
  group_by(genre) %>%
  summarize(test = list(shapiro.test(avg_rating)),
            pvalue = test %>% map_dbl("p.value"),
            normality = ifelse(pvalue > 0.05, "normal", "not normal"))
# For all but two categories the assumption of normality is violated. Therefore we are using the non-parametric Wilcoxon rank-sum test.
pairwise.wilcox.test(stats$avg_rating, stats$genre,
                     p.adjust.method= "BH")
# As can be seen, all comparisons, except"Action vs. Horror" and "Sci-Fi vs Action" have a significant p-value <0.05 after Benjamini-Hochberg multiple-testing correction. Assuming no confounders, we conclude that genre has a significant effect on movie ratings.

# To compute a coefficient for how much any genre influences the rating, we will filter for genre assignments to contain each of the available genres (except for the rarely attributed "IMAX") separately and then group by movieId. I.e. if a movie is categorized as both "Fantasy" and "Action", it will contribute to the average ratings within both "Fantasy" and "Action" genres. We then a) subtract the average of all (averaged per movie) movie ratings in the train dataset (the global mean, giving equal weight to each movie) from the category average (of averages per movie) to obtain a genre coefficient g.

global_mean <- train %>% group_by(movieId) %>% summarize(movie_avg = mean(rating)) %>% ungroup() %>% 
  summarize(global_mean = mean(movie_avg)) %>% pull(global_mean)
            
genre_cat <- setdiff(levels(avg_genre_rating$genre), c("IMAX") )

gs <- sapply(genre_cat, function(x){
train %>% filter(str_detect(genres, x)) %>% 
    group_by(movieId) %>% 
    summarize(movie_avg = mean(rating), movieId = movieId[1]) %>% 
    ungroup() %>% 
    summarize(g = mean(movie_avg - global_mean), genre = x, lower = g - 1.96*sd(movie_avg)/sqrt(n()), upper = g + 1.96*sd(movie_avg)/sqrt(n()), n = n())
}
)

g <- gs %>% unlist() %>% data.frame(genre = .[seq(2,90,5)], g = as.numeric(.[seq(1,90,5)]), lower = as.numeric(.[seq(3,90,5)]), upper = as.numeric(.[seq(4,90,5)]), n= as.numeric(.[seq(5,90,5)])) %>% select(-1) %>% slice(1:18)

g %>% mutate(genre = reorder(genre, g)) %>% 
  ggplot(aes(genre,g, ymin=lower, ymax=upper))+
  geom_point(aes(size=n),  alpha=0.5)+
  scale_size(range =c(0.3,3))+
  geom_errorbar()+
  geom_hline(yintercept = 0.0)+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  ylab("Genre residual")

ggsave("figs/genre-any_vs_rating.pdf")


saveRDS(g, "coeff/genre-g.rds")
# As can be seen from the last plot, horror movies have the most negative influence on a movie rating, while documentaries and war movies have the most favorable effect on movie ratings. It can also be seen that the ranking of the categories perfectly matches the ranking of genres when we filtered for movies that had only one movie assigned to them. Therefore, it is safe to assume that using combinations of genres to deduct the influence of any single genre is reliable enough.


# Let' s see how well the genre can predict outcome ratings. First we define the loss function which is going to be the RMSE.

# Loss function #

RMSE <- function(true_y, predicted_y){
  sqrt(mean((true_y - predicted_y)^2))
}

# Then we use the simplest predictor agaist which we will compare the success of the prediciton optimization.

#1. global mean as sole predictor
global_mean_rmse <- RMSE(test$rating, global_mean)

#2. Now we will add the genre effect to the global_mean. 

# First we define a vector with the genres k and their values g_k.

genre_values <- g$g
names(genre_values) <- g$genre

# Then we define a function to calculate the averages of values for each genre category in test$genres vector strings and.


genre_sum <- function(s) {
    categories <- strsplit(s, "\\|")[[1]]
  # Use the genre_values vector to look up the values for each category and the genre_size vectors the genre sizes.
  values <- genre_values[categories]
    # Sum the regularized values
  sum(values, na.rm = TRUE)/length(values)
}

# Apply the function to each string in the test data set genres vector and store the results
test_genre_sums <- sapply(test$genres, genre_sum_regul)
# Add a g_k column to the test data set, but also remove the names of the vector
test <- test %>% mutate(g_k = unname(test_genre_sums))
# Calculate the RMSE on the test set.
genre_model_rmse <- RMSE(test$rating, (global_mean + test$g_k))

global_mean_rmse - genre_model_rmse


#Let's add the g_k values to our train set as well.
train_genre_sums <- sapply(train$genres, genre_sum)
train <- train %>% mutate(g_k = unname(train_genre_sums))
train_genre_model_rmse <- RMSE(train$rating, (global_mean + train$g_k))

global_mean_rmse - train_genre_model_rmse

# We see that our loss is decreased by 0.014 which is a small improvement.

### RELEASE YEAR ###

# Next we want to check if there is any correlation between the year of release on the average per movie ratings. It might be that old movies were particularly entertaining or there was a peak in quality at some point, or maybe modern movies are the pinnacle of cinematography.

# Let's first extract the year from the movie title (-> year), clean up the title by removing the year, create a rating date from the timestamp (-> rating_date), extract the year of the rating (-> rating_year) and calculate an interval between the year of the rating and the year of release (-> interval) to acknowledge the fact, that highly rated movies might associate with longer intervals as they can be viewed as those which stood the test of time.

train_years <- train %>% mutate(year = parse_number(str_extract(title, "\\(\\d{4}\\)$")), 
                                title = str_replace(title, "\\s\\(\\d{4}\\)$", ""),
                                rating_date = as_datetime(timestamp), 
                                rating_year = year(rating_date),
                                interval = rating_year - year) 
test_years <- test %>% mutate(year = parse_number(str_extract(title, "\\(\\d{4}\\)$")), 
                                title = str_replace(title, "\\s\\(\\d{4}\\)$", ""),
                                rating_date = as_datetime(timestamp), 
                                rating_year = year(rating_date),
                                interval = rating_year - year) 

p1 <- train_years %>% group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), year = year[1]) %>%
  ggplot(aes(year, avg_rating)) +
  geom_jitter(alpha=0.2, size=0.5)+
  geom_smooth(method="loess", span=0.1, method.args=list(degree=1))+
  ylab("Average movie rating")+
  xlab("Release year")
p1
# We acknowledge that only a few movies were released before 1930s. However, the general trend is that old movies have higher average ratings. Additionally, there seems to be a trough in movie ratings in the 1980s. Finally, there have been many more movies released since the 1990s including the worst rated ones. It seems reasonable to consider an effect of the release year. However, we have to keep in mind that old movies might be sought after by movie enthusiasts and be among the subset of movies that have endured the test of time and are still being watched. Other old movies might be of much lower quality. Recommending an old movie just because it is old is seems therefore not reasonable. Nevertheless, we can only recommend movies that are contained in the train set, therefore we might consider this "survival" effect as a legit parameter.

# Another way to look at the time effect is to observe the interval between release date and movie rating. We might expect, that longer intervals associate with better ratings, since such movies have survived oblivion.
p2 <- train_years %>% group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), interval = interval[1]) %>%
  ggplot(aes(interval, avg_rating)) +
  geom_jitter(alpha=0.2, size=0.5)+
  geom_smooth(method="loess", span=0.3, method.args=list(degree=1))+
  ylab("Average movie rating")+
  xlab("Interval (years) between rating and release year")
p2
grid.arrange(p1, p2, nrow=1)

pdf("figs/movieage_vs_rating.pdf")
grid.arrange(p1, p2, nrow=1)
dev.off()

#Very similarly, we observe a time effect. It seems that the movies from the 1990s-2000s are condensed into an interval of 5 years, likely due to most reviews of these movies being released shortly after release date and the upcoming of the internet in the 1990s. There is a trough in ratings for an interval of 10 years and an increased approval rating the further back the release date is. Eventually, we want to stick with the release year predictor, since we want to avoid the condensation effects observed in the avg_rating vs. interval plot.


# Next, we obtain the number of movies per year to be able to use regularization if required later.
movies_pa <- train_years %>% group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), year = year[1])%>% group_by(year) %>% summarize(n_y = n())

# Let's obtain the per year estimates with the loess function and optimize span
span <- seq(0.05, 0.30, 0.05)
rmses_age_loess <- sapply(span, function(s){
loess_age <- train_years %>% group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), year = year[1], g_k=g_k[1], rating_residual = avg_rating - global_mean - g_k) %>% loess(rating_residual ~ year, ., span=s, degree=1)

data <- test_years %>% group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), year = year[1], g_k=g_k[1], rating_residual = avg_rating - global_mean - g_k)

r_hat <- predict(loess_age, newdata=data)

RMSE(data$avg_rating, (r_hat + data$g_k + global_mean))
}
)
plot(span, rmses_age_loess)

# We will use a span of 0.1
loess_age <- train_years %>% group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), year = year[1], g_k=g_k[1], rating_residual = avg_rating - global_mean - g_k) %>% loess(rating_residual ~ year, ., span=0.1, degree=1)
# Let's obtain the per year residuals yr_v
yr_v <- predict(loess_age, min(train_years$year):max(train_years$year))


# Let's now construct a function that pulls the yr_v residuals and adds them to the global mean and genre residuals and try to regularize the model with with the movies_released_per year parameter.

names(yr_v) <- movies_pa$year
movies_pa_vec <- movies_pa$n_y
names(movies_pa_vec) <- movies_pa$year

# Then we define a function to assign a year residual to each each year in test$year string vector and regularize..
l <- seq(0,10000, 2500)
year_reg_genre_model_rmse <- sapply(l, function(l){
   
# Use the yr_v vector to look up the values for each year and the movies_pa vector to estimate yearly movie cohorts
  values <- yr_v[as.character(test_years$year)]
  sizes <- movies_pa_vec[as.character(test_years$year)]
  values <- unname(values)
  sizes <- unname(sizes) #Add a yr_v_reg column to the test data set, and regularize with the tuning parameter l
test <- test_years %>% mutate(yr_v_reg = values*sizes/(sizes+l), yr_v = values)
# Calculate the RMSE on the test set.
year_reg_genre_rmse <- RMSE(test$rating, (global_mean + test$yr_v_reg + test$g_k))
}
)
year_reg_genre_model_rmse
plot(l, year_reg_genre_model_rmse)
min(year_reg_genre_model_rmse)
global_mean_rmse - min(year_reg_genre_model_rmse)

# the year factor does not improve the model, it even disturbs it. This is likely due to a too wide range of movie ratings each year.

### RATING FREQUENCY ###

# The last time effect we want to consider is the amount of ratings, normalized to year of existence (last review date minus release year), and to the size of the database, i.e. the total amount of reviews,that a movie has received (we will call it "ratings_pa"). This will allow to use this parameter when analyzing other datasets with different sizes. The measure of how often a movie is rated allows the extrapolation to how often it was watched and therefore how popular it is, as well as how reliable the ratings are (considering the central limit theorem here). We will take 2009 as the cutoff year, since 2008 is the year of the last released movie in our data set and we don't want to divide by 0.

range(train_years$year) # Range of release year between 1915 and 2008. Use 2009.

p3 <- train_years %>% group_by(movieId) %>% 
  summarize(ratings_pa = n()/(2009-year[1])/dim(train)[1]*10^6, avg_rating=mean(rating)) %>%
  ggplot(aes(sqrt(ratings_pa), avg_rating)) +
  geom_jitter(alpha=0.2, size=0.5)+
  geom_smooth(method="loess", span=0.3, method.args=list(degree=1))+
  ylab("Average movie rating")+
  xlab("Sqrt(Normalized ratings per year)")
p3
ggsave("figs/ratings-pa_vs_rating.pdf")
#Let's check for correlation, but filtering out small values first

train_ratings_pa <- train_years %>% group_by(movieId) %>% 
  summarize(ratings_pa = n()/(2009-year[1])/dim(train)[1]*10^6, avg_rating=mean(rating)) 

train_ratings_pa %>% filter(ratings_pa > quantile(ratings_pa, 0.10)) %>% summarize(cor = (cor(sqrt(ratings_pa), avg_rating, method="spearman")))

#The pearson correlation coefficient is r = 0.166

# To better visualize, let's construct intervals of the rating per year and the associated average rating minus the global mean
range(sqrt(train_ratings_pa$ratings_pa))

train_ratings_pa %>% mutate(strata = cut(sqrt(train_ratings_pa$ratings_pa), breaks = seq(0, 16.4, by = 1.64))) %>%
  group_by(strata) %>% 
  summarize(mean_rating = mean(avg_rating - global_mean), 
            lower = mean_rating - 1.96*sd(avg_rating)/sqrt(n()), 
            upper = mean_rating + 1.96*sd(avg_rating)/sqrt(n()), n= n() ) %>% 
  ggplot(aes(strata, mean_rating, ymin=lower, ymax=upper))+
  geom_point(aes(size=n), alpha=0.7)+
  geom_errorbar(width=0.5)+
  theme(axis.text.x = element_text(hjust=1, angle=90))+
  ylab("Ratings per year Residual")+
  xlab("Strata of Sqrt(Norm. ratings per year)")
ggsave("figs/ratings-pa-strat_vs_rating.pdf")

# This plot shows a clear trend between rating rate and rating score. The less reviews a movie gets, the noisier and wide-spread the rating score distribution is. The trend in this plot is very clear with frequently reviewed movies receiving higher scores. Ratings per year might allow for regularization of other predictors, penalizing rarely viewed and rated, therefore likely obscure, movies. We can also conclude that higher ratings associate with higher rating rate, hence also higher number of viewers. Therefore a higher rating indicates audience interest. Movies that people enjoy to watch receiver higher scores and we should recommend movies that are likely to be rated highly by a user.

# Let's build a loess predictor for rating score vs. ratings per year, a procedure equivalent to the one used in the "year" parameter section

# Let's obtain the per rating_rate estimates with the loess function and optimize span
span <- seq(0.1, 1.0, 0.1)
rmses_ratingrate_loess <- sapply(span, function(s){
  
  loess_ratingrate <- train_years %>% 
    group_by(movieId) %>% 
    summarize(ratings_pa = n()/(2009-year[1])/dim(train)[1]*10^6, avg_rating=mean(rating), g_k=g_k[1], rating_residual = avg_rating - global_mean - g_k) %>% 
    loess(rating_residual ~ ratings_pa, ., span=s, degree=1)
  
    data <- test_years %>% 
      group_by(movieId) %>% 
      summarize(ratings_pa = n()/(2009-year[1])/dim(test)[1]*10^6, avg_rating=mean(rating), g_k=g_k[1], rating_residual = avg_rating - global_mean - g_k)
    
  r_hat <- predict(loess_ratingrate, newdata=data)
  
  RMSE(data$avg_rating, (r_hat + data$g_k + global_mean))
}
)
plot(span, rmses_ratingrate_loess)

# We will use a span of 0.2
loess_ratingrate <- train_years %>% group_by(movieId) %>% 
  summarize(ratings_pa = n()/(2009-year[1])/dim(train)[1]*10^6, avg_rating=mean(rating), g_k=g_k[1], rating_residual = avg_rating - global_mean - g_k) %>% 
  loess(rating_residual ~ ratings_pa, ., span=0.2, degree=1)

# Let's obtain the per year ratings residuals rate_pa_w over a range of observed values. We substitute 0 with 0.01 to obtain a value.
rate_pa_w <- predict(loess_ratingrate, c(0.01, 1:round(max(train_ratings_pa$ratings_pa))))

# Let's now construct a function that pulls the rate_pa_w residuals and adds them to the global mean and genre residuals and try to regularize the model with a movies with a rating_pa x parameter.

ratings_pa_strata <- train_ratings_pa %>% mutate(ratings_pa_int = round(ratings_pa)) %>% group_by(ratings_pa_int) %>% summarize(n=n()) 

ratings_pa_strata_vec <- ratings_pa_strata$ratings_pa_int
ratings_pa_int_sizes <- ratings_pa_strata$n
names(ratings_pa_int_sizes) <- ratings_pa_strata$ratings_pa_int
names(rate_pa_w) <- 0:266

# Let's add the rating_pa to the test data set
test <- test %>% left_join(train_ratings_pa, by="movieId")

# Then we define a function to assign a rating_pa residual to each each rounded stratum w in test$ratings_pa vector and regularize..
l <- seq(0, 100, 10)
rating_pa_reg_genre_model_rmse <- sapply(l, function(l){
    # Use the rate_pa_w vector to look up the values for each rating_pa stratum and the ratings_pa_int_sizes vector to estimate yearly movie rating size for that stratum for regularization
  values <- rate_pa_w[as.character(round(test$ratings_pa))]
  sizes <- ratings_pa_int_sizes[as.character(round(test$ratings_pa))]
  values <- unname(values)
  sizes <- unname(sizes) #Add a rate_pa_w_reg column to the test data set, and regularize with the tuning parameter l
  test <- test %>% mutate(rate_pa_w_reg = values*sizes/(sizes+l))
  # Calculate the RMSE on the test set.
  rating_pa_reg_genre_rmse <- RMSE(test$rating, (global_mean + test$rate_pa_w_reg + test$g_k))
}
)
rating_pa_reg_genre_model_rmse
plot(l, rating_pa_reg_genre_model_rmse)
min(rating_pa_reg_genre_model_rmse)
global_mean_rmse - min(rating_pa_reg_genre_model_rmse)

# Regularization does not benefit for the very reason that movies with 0-1 ratings per year make up more than 50% of all movies, but have a negative ratings_pa residual, which likely is a generally good rule of thumb to rate rarely viewed, therefore likely obscure, movies worse than average.

# We therefore simplify the model and obtain a RMSE value without any regularization:

  values <- rate_pa_w[as.character(round(test$ratings_pa))]
  values <- unname(values)
  test <- test %>% mutate(rate_pa_w_reg = values)
  # Calculate the RMSE on the test set.
  rating_pa_reg_genre_rmse <- RMSE(test$rating, (global_mean + test$rate_pa_w_reg + test$g_k))
  rating_pa_reg_genre_rmse
  global_mean_rmse - rating_pa_reg_genre_rmse
  
#We will save the ratings_pa residuals in a separate object, as they benefit in reducing the RMSE by additional 0.054!
# we need to create a vector for all possible values of rating_pa and fill in the available stratum sizes
vec = rep(0,267)
names(vec) = 0:266
vec[names(ratings_pa_int_sizes)] <- ratings_pa_int_sizes

rate_pa_w_df <- predict(loess_ratingrate, c(0.01, 1:round(max(train_ratings_pa$ratings_pa))), se=TRUE)
rate_pa_w_conf <- data.frame(rate_pa = 0:266, rate_pa_w = rate_pa_w_df[[1]], lower = rate_pa_w -1.96 * rate_pa_w_df[[2]], upper = rate_pa_w +1.96 * rate_pa_w_df[[2]], n = vec)

rate_pa_w_conf %>% ggplot(aes(rate_pa, rate_pa_w))+
  geom_line(col="red", lwd=1)+
  geom_line(aes(rate_pa, lower), col="blue", lty=2)+
  geom_line(aes(rate_pa, upper), col="blue", lty=2)+
  ylab("Estimated rating residual")+
  xlab("Norm. ratings per year")

ggsave("figs/rating_pa_loess.pdf")

saveRDS(rate_pa_w_conf, "coeff/rating_pa_w.rds")

### MOVIE EFFECTS ###

# Now we want to consider the primary cause and effect. How a movie, the subject of the rating, itself influences the rating. Are there movie effects? Let's see how the values are distributed.
train %>% group_by(movieId) %>% summarize(avg_rating = mean(rating), n = n()) %>%
  ggplot(aes(avg_rating))+
  geom_histogram()

mean(train_ratings_pa$avg_rating >= global_mean)
mean(train_ratings_pa$avg_rating < global_mean)

# Movie ratings show a slighty skewed right-sided distribution, with 55% of all movie ratings resulting in above average ratings.
quantile(train_ratings_pa$avg_rating, probs = seq(0.1, 1.0, 0.1))

train %>% group_by(movieId) %>% summarize(avg_rating = mean(rating), n = n()) %>% mutate(movieId = reorder(movieId, avg_rating)) %>%
  ggplot(aes(movieId, avg_rating))+
  geom_point(aes(size=n))+
  theme(axis.text.x = NULL)
