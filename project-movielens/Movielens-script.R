### Make sure to go to Tools -> Global Options -> Code -> tick "Soft-wrap R source files" to see comments wrapped ###
### THE PROJECT IS AVAILABLE UNDER https://github.com/dblnk/Data-Science/tree/master/project-movielens ###
### THE DEVELOPING OF THE MODEL CONSISTS OF STEPS 1-10 -> navigate if necessary through the dropwdown menu ###
### All effects are understood as effect estimates. The "hat" notation of estimates is therefore omitted ###
### MAKE SURE TO RUN THE CODE IN LINES 13-115 TO GENERATE "edx.rds", "final_holdout_test.rds", "train.rds" and "test.rds" FILES INTO A data/ DIRECTORY ###

############### START OF INITIAL CODE PROVIDED BY PROF. IRIZARRY AND HIS TEAM ############### 

# Create edx and final_holdout_test sets 

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

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

############### END OF INITIAL CODE PROVIDED BY PROF. IRIZARRY AND HIS TEAM ############### 

#We save the data sets in order to load them directly in the future, when restarting the session.

saveRDS(edx, "data/edx.rds")
saveRDS(final_holdout_test, "data/final_holdout_test.rds")
rm(edx, final_holdout_test)

edx <- readRDS("data/edx.rds")

if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# First we split the edx data into a training (80%) and a validation test set (20%). We will use the training set to derive our model and test the model performance on the test set.
set.seed(1998)
test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
train <- edx[-test_index, ]
temp <- edx[test_index, ]

# verify if all movieId and userId values are present in both train and test sets.

test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Since there are some removed rows, add the removed rows from test set back into the training set.
removed <- anti_join(temp, test)
train <- rbind(train, removed)

#Again, we save the split data sets in order to load them directly in the future, when restarting the session.
saveRDS(train, "data/train.rds")
saveRDS(test, "data/test.rds")
rm(edx, final_holdout_test)

train <- readRDS("data/train.rds")
test <- readRDS("data/test.rds")

rm(test_index, temp, removed)

options(digits=7)

############### STEP 1: MOVIE EFFECTS ###############

# Since the quality of a movie should have the largest effect on its rating and we need to make a judgement if we want to recommend a particular movie, we will first derive a measure of how a movie, the subject of the rating, influences the review score. Naively put, are there movie effects? Let's see how the values are distributed after averaging the ratings for each movie.

# We calculate a global average by first averaging the ratings for each movie and then calculate the overall mean value of the movie averages.
global_mean <- train %>% group_by(movieId) %>% summarize(movie_avg = mean(rating)) %>% ungroup() %>% 
  summarize(global_mean = mean(movie_avg)) %>% pull(global_mean)

  ds_theme_set()
  
p1 <- train %>% group_by(movieId) %>% summarize(avg_rating = mean(rating), n = n()) %>%
  ggplot(aes(avg_rating))+
  geom_histogram()+
  geom_vline(xintercept=global_mean, color = "blue")+
  geom_text(aes(x=2.5, y=1200, label="Overall average"), color="blue")+
  xlab("Average movie rating")+
  ylab("Count")+
  ggtitle("Distribution of movie ratings")+
  theme(plot.title = element_text(hjust=0.5))

temp<- train %>% group_by(movieId) %>% summarize(avg_rating = mean(rating), n = n())
paste0("Movie ratings above overall average: ", percent (mean(temp$avg_rating >= global_mean), accuracy=0.1))

# Movie ratings show a sightly skewed right-sided distribution, with 55% of all movie ratings resulting in above average ratings.

# How many reviews do the movies receive?
p2 <- train %>% group_by(movieId) %>% summarize(n = n()) %>%
  ggplot(aes(sqrt(n)))+
  geom_histogram()+
  geom_vline(aes(xintercept=median(sqrt(n))), color = "blue")+
  geom_text(aes(x=40, y=3000, label=paste0("Median = ",signif(median(sqrt(n)),digits=3))), color="blue")+
  xlab("Sqrt(ratings per movie)")+
  ylab("Count")+
  ggtitle("Number of ratings per movie")+
  theme(plot.title = element_text(hjust=0.5))

grid.arrange(p1, p2, nrow =1)

pdf("figs/movie-ratings.pdf", width = 10, height = 6)
grid.arrange(p1, p2, nrow =1)
dev.off()
# It is  obvious that most of the movies received less than 10 ratings. Therefore estimating their average rating from the opinion of only a few users might not be very robust.


# Now, let's start building our model by adding a movie effect m_i for movie i consisting of its average rating minus the global mean and regularize the effect by the amount of reviews that a movie has received to avoid over- or underestimation of rarely rated movies. 
# Our model is currently the following:
# $R_i = µ + m_i + €_i$, with R_i being the expected rating of movie i, µ the global mean, m_i the movie effect for movie i and €_i the residual for movie i.

# Next, we define the loss function by which we will measure the goodness of fit for our model. We are going use the root mean squared error (RMSE).

# Defining the RMSE loss function
RMSE <- function(true_r, predicted_r){
  sqrt(mean((true_r - predicted_r)^2))
}

# Then we compute the goodness of fit of the simplest predictor, the mean of all movie ratings. We will use this value to evaluate the success of our model optimization.

# Global mean as sole predictor of the ratings in the test set
global_mean_rmse <- RMSE(test$rating, global_mean)
paste0("RMSE for global mean as predictor: ",signif(global_mean_rmse, digits=7))

# Movie effect
# We calculate the movie effect by subtracting the global_mean from the average movie rating. We also define a tuning parameter l to penalize predictions of rarely reviewed movies. We are probing a range of l and examine which value of l minimizes the RMSE in the test set.
l <- seq(0, 10, 1)
movie_rmses <- sapply(l, function(l){
  m_i <- train %>% group_by(movieId) %>% 
    summarize(m_i = sum(rating - global_mean)/(n() + l))
  r_hat <- test %>% # r_hat is our new predicted rating
    left_join(m_i, by="movieId") %>% 
    mutate(r_hat = global_mean + m_i) %>% pull(r_hat)
  RMSE(test$rating, r_hat)
})
plot(l,movie_rmses)
l[which.min(movie_rmses)]
min(movie_rmses)
global_mean_rmse - min(movie_rmses)
# A tuning parameter l of 5 seems to be optimal. And our RMSE is decreased by 0.164! Now it is 0.943.

# Now we will add the movie effects to our train and test data sets, using the tuning parameter of 5.
m_i <- train %>% group_by(movieId) %>% 
  summarize(m_i = sum(rating - global_mean)/(n() + 5))
train <- train %>% left_join(m_i, by="movieId")
test <- test %>% left_join(m_i, by="movieId")

movie_rmse <-RMSE(test$rating, (global_mean + test$m_i))

paste0("RMSE with movie effects: ",signif(movie_rmse, digits=7))

############### STEP 2: USER effects ###############

# Now we want to consider the effects of subjective rating based on user identity. How does the perception of the individual influence their judgement? Are there user effects? Let's see how the values of the average user rating and the amount of ratings each user has left are distributed.
temp <- train %>% group_by(userId) %>% summarize(avg_user_rating = mean(rating), n = n())

#The average rating per user
p3 <- temp %>%
  ggplot(aes(avg_user_rating))+
  geom_histogram()+
  geom_vline(xintercept=global_mean, color = "blue")+
  geom_text(aes(x=2.0, y=10000, label="Overall movie average"), color="blue")+
  xlab("Average user rating")+
  ylab("Count")+
  ggtitle("Distribution of user ratings")+
  theme(plot.title = element_text(hjust=0.5))

# How many ratings does each user distribute
p4 <- temp %>%
  ggplot(aes(n))+
  scale_x_continuous(trans="log2", breaks=c(4,8,16,32,64,128,256,512,1024,2048))+
  geom_histogram()+
  geom_vline(aes(xintercept=median(n)), color = "blue")+
  geom_text(aes(x=120, y=6500, label=paste0("Median = ",signif(median(n),digits=3))), color="blue")+
  xlab("Ratings per user)")+
  ylab("Count")+
  ggtitle("Number of ratings per user")+
  theme(plot.title = element_text(hjust=0.5))

grid.arrange(p3, p4, nrow=1)

pdf("figs/user-ratings.pdf", width = 10, height = 6)
grid.arrange(p3, p4, nrow =1)
dev.off()

paste0("Average user ratings above overall movie average: ", percent (mean(temp$avg_user_rating >= global_mean), accuracy=0.1))
#Interestingly, the average ratings per user are very favorable, with 85% of users rating above the global mean on average. We might therefore be able to identify the subset of users that are more critical of movies in general and account for their demanding taste, but also those who enjoy movies much more than the average. Half of the users have left less than 50 ratings. This implies that we might be able to make a robust prediction of the individual user tendency.

#At last, let's take a quick look how the average user rating relates to the amount of ratings they have distributed at the top and bottom 15% ranges.

quants <- quantile(temp$avg_user_rating, probs = c(0.15, 0.85))

p5 <- temp %>% filter(avg_user_rating < quants[1] | avg_user_rating > quants[2]) %>% 
  ggplot(aes(avg_user_rating, sqrt(n), color=sqrt(n)))+
  geom_point(aes(size=n), alpha=0.2)+
  scale_color_gradient(low="darkblue", high="magenta")+
  xlab("Average user rating")+
  ylab("Sqrt(number (n) of ratings per user)")+
  ggtitle("Number of ratings per user vs. average rating per user")+
  theme(plot.title = element_text(hjust=0.5))
p5

pdf("figs/user-ratings_vs_n.pdf")
p5
dev.off()

# We can see that the more extreme raters usually write only relatively few reviews and that the frequent raters have average ratings much closer to the global mean. Therefore we also need to penalize low rating rate by regularization of the user effect, as otherwise we might under- or overestimate the quality of the movie.

# Now let's expand our movie_effect model by considering a user effect u_j consisting of the average rating residual of this user and regularize the effect by the amount of reviews that this user has contributed to our train set.
# Our updated model is thus the following:
# $R_i,j = µ + m_i + u_j + €_i,j$, with R_i,j being the expected rating of user j for movie i , µ the global mean, m_i the movie effect for movie i, u_j the user effect for user j and €_i,j the residual for movie i and user j.

l <- seq(0, 10, 1)
user_movie_rmses <- sapply(l, function(l){
  u_j <- train %>% group_by(userId) %>% 
    summarize(u_j = sum(rating - global_mean - m_i)/(n() + l))
  r_hat <- test %>% 
    left_join(u_j, by="userId") %>% 
    mutate(r_hat = global_mean + m_i + u_j) %>% pull(r_hat)
  RMSE(test$rating, r_hat)
})

plot(l,user_movie_rmses)
l[which.min(user_movie_rmses)]
min(user_movie_rmses)
movie_rmse - min(user_movie_rmses)

# A tuning parameter l of 5 is optimal to minimize the RMSE by another 0.0777 points! Now it is 0.8654.

# Now we will add the user effects to our train and test data sets, using the tuning parameter of 5.
u_j <- train %>% group_by(userId) %>% 
  summarize(u_j = sum(rating - global_mean - m_i)/(n() + 5))

train <- train %>% left_join(u_j, by="userId")

test <- test %>% left_join(u_j, by="userId")

#Current RMSE considerung user and movie effects to explain the residuals of the predicted rating (approximated by the global mean):
user_movie_rmse <- RMSE(test$rating, (global_mean + test$m_i + test$u_j))

paste0("RMSE with movie and user effects: ",signif(user_movie_rmse, digits=7))


###############STEP 3: TIME EFFECTS ###############
# To consider time-dependent effects we first need to extract the year of release from the movie title (-> year), we will also clean up the title by removing the year and create a rating date from the timestamp (-> rating_date), in both training and test data sets.

train <- train %>% mutate(year = parse_number(str_extract(title, "\\(\\d{4}\\)$")), 
                          title = str_replace(title, "\\s\\(\\d{4}\\)$", ""),
                          rating_date = as_datetime(timestamp)) 
test <- test %>% mutate(year = parse_number(str_extract(title, "\\(\\d{4}\\)$")), 
                        title = str_replace(title, "\\s\\(\\d{4}\\)$", ""),
                        rating_date = as_datetime(timestamp)) 

# We start by taking into consideration how many ratings each movie has received per year (last review date minus release year), and divide this annual rating rate by the size of the database, i.e. the total amount of reviews (we will call this feature "ratings_pa"). This will allow using this parameter when analyzing other data sets with different sizes. The measure of how often a movie is rated allows the extrapolation to how often it is being watched and therefore how popular it is, as well as how reliable the ratings are (considering the central limit theorem here).
range <- range(train$year)
range
# We will take 2009 as the cutoff year, since 2008 is the year of the last released movie in our data set and we cannot divide by 0. We will multiply by a scaling factor of 10^6 to obtain reasonable decimals.

# Calculating rating frequency and store in a new data frame
train_ratings_pa <- train %>% group_by(movieId) %>% 
  summarize(ratings_pa = n()/(range[2]+1-year[1])/dim(train)[1]*10^6, avg_rating=mean(rating))

# Let's visualize if our assumption seems reasonable by plotting our current rating residual against the rating frequency.
p6 <- train %>% group_by(movieId) %>% 
  summarize(ratings_pa = n()/(range[2]+1-year[1])/dim(train)[1]*10^6, rating_residual = mean(rating - global_mean - m_i - u_j), n=n()) %>%
  ggplot(aes(sqrt(ratings_pa), rating_residual)) +
  geom_jitter(aes(color=sqrt(n)), alpha=0.4)+
  geom_smooth(method="loess", span=0.3, method.args=list(degree=1))+
  ylab("Rating residual (post movie & user effects)")+
  xlab("Sqrt(Normalized ratings per year)")+
  ggtitle("Movie residuals vs. rating frequency")+
  geom_hline(yintercept=0, color="black", alpha=0.7)+
  scale_color_gradient(low="grey", high="blue")+
  labs(color = "Sqrt(n p. movie)")+
  theme(plot.title = element_text(hjust=0.5))
p6

pdf("figs/ratings-pa_vs_residuals.pdf")
p6
dev.off()

# We observe that movies with low rating frequencies tend to have positive residuals and we might improve our estimate by accounting for the ratings_pa feature.

# Our updated model is thus the following:
# $R_i,j = µ + m_i + u_j + rate_pa_i + €_i,j$, with R_i,j being the expected rating of user j for movie i , µ the global mean, m_i the movie effect for movie i, u_j the user effect for user j, rate_pa_i the rating frequency effect for movie i, €_i,j the residual for movie i and user j.

# Let's build a loess predictor for the residual vs. ratings per year, and optimize span, using RMSE on the test set as an indicator of how the span affects our estimates.
span <- seq(0.05, 0.5, 0.05)
rmses_ratingrate_loess <- sapply(span, function(s){
  
  loess_ratingrate <- train %>% 
    group_by(movieId) %>% 
    summarize(ratings_pa = n()/(range[2]+1-year[1])/dim(train)[1]*10^6, avg_rating=mean(rating), m_i=m_i[1], u = mean(u_j), rating_residual = avg_rating - global_mean - m_i - u) %>% 
    loess(rating_residual ~ ratings_pa, ., span=s, degree=1)
  
  data <- test %>% 
    group_by(movieId) %>% 
    summarize(ratings_pa = n()/(range[2]+1-year[1])/dim(test)[1]*10^6, avg_rating=mean(rating), m_i=m_i[1], u = mean(u_j), rating_residual = avg_rating - global_mean - m_i - u)
  
  r_hat <- predict(loess_ratingrate, newdata=data)
  
  RMSE(data$avg_rating, (r_hat + data$m_i + data$u + global_mean))
}
)
plot(span, rmses_ratingrate_loess)
span[which.min(rmses_ratingrate_loess)]

# We will use a span of 0.2 to compute the loess function

loess_ratingrate <- train %>% group_by(movieId) %>% 
  summarize(ratings_pa = n()/(range[2]+1-year[1])/dim(train)[1]*10^6,  avg_rating=mean(rating), m_i=m_i[1], u = mean(u_j), rating_residual = avg_rating - global_mean - m_i - u) %>% 
  loess(rating_residual ~ ratings_pa, ., span=0.2, degree=1)

# Let's obtain the per year ratings effect rate_pa_i for a movie i over a range of observed values, rounded to integers. We substitute 0 with 0.01 to obtain a non-NA value.
rate_pa_i <- predict(loess_ratingrate, c(0.01, 1:round(max(train_ratings_pa$ratings_pa))))

# Now we construct a function that pulls the rate_pa_i residual effects and adds them to the global mean, movie and user effects and try to regularize the model with a rating_pa_strata size parameter, reflective of how many movies fall into each integer of the rounded rating_pa feature.

ratings_pa_strata <- train_ratings_pa %>% mutate(ratings_pa_int = round(ratings_pa)) %>% group_by(ratings_pa_int) %>% summarize(n=n()) 

# Assign all possible strata to the values in rate_pa_i as vector entry names.
names(rate_pa_i) <- 0:round(max(train_ratings_pa$ratings_pa))
# Construct a vector storing the actual strata sizes and name the entries with the observed integers
ratings_pa_int_sizes <- ratings_pa_strata$n
names(ratings_pa_int_sizes) <- ratings_pa_strata$ratings_pa_int

# Let's add the rating_pa feature to the train  data sets.
train <- train %>% left_join(train_ratings_pa, by="movieId")

# Then we define a function to assign a rating_pa residual to each stratum w in the train$ratings_pa vector and examine if regularization by stratum size has any impact on our loss.
# Choose the tuning parameter
l <- seq(0, 2.5, 0.25)
# Build the function
rating_pa_model_rmses <- sapply(l, function(l){
  # Use the named rate_pa_i vector to look up the values for each rating_pa stratum and the ratings_pa_int_sizes vector to estimate yearly movie rating size for that stratum for regularization
  values <- rate_pa_i[as.character(round(train_ratings_pa$ratings_pa))]
  sizes <- ratings_pa_int_sizes[as.character(round(train_ratings_pa$ratings_pa))]
  # unname the values and sizes vectors
  values <- unname(values)
  sizes <- unname(sizes) 
  #Add a rate_pa_i column to the train data set, regularize with the tuning parameter l and then join it to the test set.
  train_bymovie <- train %>% group_by(movieId) %>% summarize(ratings_pa=ratings_pa[1]) %>% mutate(rate_pa_i = values*sizes/(sizes+l))
  
  test <- train_bymovie %>% select(movieId, rate_pa_i) %>% right_join(test, by="movieId", multiple="all")
  # Calculate the RMSE on the test set.
  rating_pa_reg_rmses <- RMSE(test$rating, (global_mean + test$rate_pa_i + test$m_i + test$u_j))
  rating_pa_reg_rmses
}
)
rating_pa_model_rmses
plot(l, rating_pa_model_rmses)
l[which.min(rating_pa_model_rmses)]
user_movie_rmse - min(rating_pa_model_rmses)

# Regularization does not benefit, for the very reason that movies with few ratings per year make up more than 50% of all movies, but have a positive ratings_pa estimate.

# We therefore simplify the model and obtain a RMSE value without any regularization:

values <- rate_pa_i[as.character(round(train_ratings_pa$ratings_pa))]
values <- unname(values)

train_rate_pa_i <- train %>% group_by(movieId) %>% summarize(ratings_pa =ratings_pa[1]) %>% mutate(rate_pa_i = values)
train <- train_rate_pa_i %>% select(movieId, rate_pa_i) %>% right_join(train, by="movieId", multiple="all")
test <- train_rate_pa_i %>% select(movieId, rate_pa_i) %>% right_join(test, by="movieId", multiple="all")

# Calculate the RMSE on the test set.
rating_pa_user_movie_rmse <- RMSE(test$rating, (global_mean + test$rate_pa_i + test$m_i + test$u_j))
paste0("RMSE with movie, user and movie rating frequency effects: ",signif(rating_pa_user_movie_rmse, digits=7))
user_movie_rmse - rating_pa_user_movie_rmse

# We managed to reduce the RMSE by 0.00018 which is a small improvement. It is now 0.86520.

rm(temp, train_rate_pa_i, loess_ratingrate, p1,p2,p3,p4,p5,p6, ratings_pa_strata, train_ratings_pa, train_users, movie_rmses, quants, range, rating_pa_model_rmses, ratings_pa_int_sizes, rmses_ratingrate_loess, values, user_movie_rmses)

############### STEP 4: MOVIE-GENRE effects ###############

# We will continue by examining if our current residuals (after accounting for movie, user and rating frequency effects) can be explained by genre.

# We want to extract the single genres that the movies are associated with. At first, we want to look at those movies that have only one genre assigned. This allows us to avoid the genre assignment ambiguities of many of the entries and explore if unambiguous, singular genre assignments have an influence on the rating outcome. 
set.seed(2018)
train %>% select(genres) %>% slice_sample(n=10)
#We see that the genres of each movie are separated by a "|" symbol. We will use the regex expression "\\|" to remove    all categories that contain any "|".

# Let's look at how the ratings are distributed across the unambiguous genre categories. We will average the ratings that each movie has received. And we will order the factor levels of genre by the ascending order of the median average rating residuals.
med_genre_rating <- train %>% 
  filter(!str_detect(genres, "\\|")) %>%
  group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), avg_residual=mean(rating-global_mean - m_i - u_j - rate_pa_i), genre = genres[1]) %>% group_by(genre) %>%
  summarize(med_genre_rating = median(avg_rating), med_genre_residual = median(avg_residual), n = n()) %>% mutate(genre = reorder(genre, med_genre_residual)) 
med_genre_rating <- med_genre_rating[-1,] # removed "(no genres listed)"

p7 <- train %>% filter(!str_detect(genres, "\\|")) %>% filter(genres != "(no genres listed)") %>%
  group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), genre = genres[1], avg_residual=mean(rating-global_mean - m_i - u_j - rate_pa_i)) %>%
  mutate(genre = factor(genre, levels=levels(med_genre_rating$genre))) %>%
  ggplot(aes(genre, avg_residual)) + 
  geom_jitter(alpha=0.2, size=1)+
  geom_boxplot(outlier.shape = NA, color="blue") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  ylab("Average residual per movie")+
  xlab("Genre (unique only)")+
  ggtitle("Residuals (post movie, user & rate_pa effects) vs. Genre")+
  geom_hline(yintercept=0, color="black")+
  theme(plot.title = element_text(hjust=0.5))
p7

pdf("figs/genre-singl_vs_residuals.pdf")
p7
dev.off()

# It is apparent from this plot that some genres, such as "Children" or "Fantasy" are filled with only a few movies, hence they are likely to be rarely the only genre attributes to a movie. Therefore, we will not be able to reliably estimate the effect of a singular genre for most genres. However, to make a general case we will have a look if there are any significant differences between categories that are sufficiently crowded (n>50), i.e. "Horror","Sci-Fi", "Action", Comedy", "Thriller", "Western", "Drama" and "Documentary".

genre_crowded <- train %>% filter(!str_detect(genres, "\\|")) %>% filter(genres != "(no genres listed)") %>%
  group_by(movieId) %>% 
  summarize(genre = genres[1]) %>%
  mutate(genre = factor(genre, levels=levels(med_genre_rating$genre))) %>%
  group_by(genre) %>% tally() %>% arrange(desc(n)) %>% filter(n>=50) %>%pull(genre)

# Create a data.frame that can be used for statistical analysis and visualization.
stats <- train %>% filter(!str_detect(genres, "\\|")) %>% filter(genres != "(no genres listed)") %>%
  group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), avg_residual=mean(rating-global_mean - m_i - u_j - rate_pa_i), genre = genres[1]) %>%
  mutate(genre = factor(genre, levels=levels(med_genre_rating$genre))) %>% filter(genre %in% genre_crowded)

# Let's plot the selected genres.
stats %>%
  ggplot(aes(genre, avg_residual)) + 
  geom_jitter(alpha=0.2, size=1)+ #we limit the number of points shown 
  geom_boxplot(outlier.shape = NA, color="blue") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  ylab("Average residual per movie")+
  xlab("Genre (unique only)")+
  ggtitle("Residuals (post movie, user & rate_pa effects) vs. Genre")+
  geom_hline(yintercept=0, color="black")+
  theme(plot.title = element_text(hjust=0.5))

ggsave("figs/genre-singl-crwd_vs_rating.pdf")

#First we check for normality
stats %>%
  group_by(genre) %>%
  summarize(test = list(shapiro.test(avg_residual)),
            pvalue = test %>% map_dbl("p.value"),
            normality = ifelse(pvalue > 0.05, "normal", "not normal"))
# the assumption of normality is violated for all genres. Therefore we are using the non-parametric Wilcoxon rank-sum test to assess statistical significane of the genre differences.
wilcox_results <- pairwise.wilcox.test(stats$avg_residual, stats$genre,
                     p.adjust.method= "BH")
# Extract p-values and save them into a data frame
p_values <- tidy(wilcox_results, p.value = "p.value") %>%
  pivot_wider(names_from = group1, values_from = p.value) %>%
  column_to_rownames(var = "group2")
# Display p-values in a table
knitr::kable(p_values, caption = "Pairwise Wilcoxon test p-values")

# As can be seen, the majority of comparisons, except for e.g. "Action vs. Horror", "Sci-Fi vs Action" or "Comedy vs. Thriller" have a significant p-value <0.05 after Benjamini-Hochberg multiple-testing correction. We conclude that genre has significant effects on current model's movie rating residuals, with some genres like Sci-Fi having lower than 0 and some like Documentary higher than 0 median values.

# To compute a quantitative term g_k for how much any genre k influences the rating, we will reiteratively filter for genre assignments to contain each of the available genres (except for the rarely attributed non-genre term "IMAX") separately and then group by movieId. I.e. if a movie is categorized as both "Fantasy" and "Action", it will contribute to the average ratings within both "Fantasy" and "Action" genres. We then a) subtract the average of all (averaged per movie) rating residuals in the train dataset from the category average (of averages per movie) to obtain a genre coefficient g_k for genre k.

# Object containing all movies and their genre assignments with average movie ratings and number of ratings
avg_mov_rating <- train %>% 
  group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), genre = genres[1], n=n())

# Obtain all possible genres from genre column of avg_mov_rating
genre_list <- str_split(avg_mov_rating$genre, "\\|")
unique_genres <- sort(unique(unlist(genre_list))) # identify unique genres and sort alphabetically
unique_genres # the vector contains a "(no genres listed)" and an "IMAX" genre. IMAX is not a genre, but a film format. Only few movies have it assigned.
avg_mov_rating %>% filter(str_detect(genre, "IMAX")) %>% tally() 
# We will get rid of "(no genres listed)" and "IMAX" genre entries.
which(unique_genres == "IMAX")
unique_genres <- unique_genres[-c(1, 13)]
 unique_genres

# Construct a function to filter the training data for each of the genre terms and calculate the average residual per genre with their 95% confidence interval. Save these in a list called gs.
gs <- sapply(unique_genres, function(x){
  train %>% filter(str_detect(genres, x)) %>% 
    group_by(movieId) %>% 
    summarize(avg_residual=mean(rating -  global_mean  - m_i - u_j - rate_pa_i)) %>%
    ungroup() %>% 
    summarize(g = mean(avg_residual), genre = x, lower = g - 1.96*sd(avg_residual)/sqrt(n()), upper = g + 1.96*sd(avg_residual)/sqrt(n()), n = n())
}
)

# Convert the resulting list gs into a data frame g with values g_k for each genre and confidence intervals
g <- gs %>% unlist() %>% data.frame(genre = .[seq(2,90,5)], g = as.numeric(.[seq(1,90,5)]), lower = as.numeric(.[seq(3,90,5)]), upper = as.numeric(.[seq(4,90,5)]), n= as.numeric(.[seq(5,90,5)])) %>% select(-1) %>% slice(1:18)

#Let's plot the genres against average rating residuals per movie
p8 <- g %>% mutate(genre = reorder(genre, g)) %>% 
  ggplot(aes(genre,g, ymin=lower, ymax=upper))+
  geom_point(aes(size=n),  alpha=0.5)+
  scale_size(range =c(0.3,3))+
  geom_errorbar()+
  geom_hline(yintercept = 0.0)+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  ylab("Average rating residual per genre")+
  xlab("Genre (multiple per movie possible)")+
  ggtitle("Residuals (post movie, user & rate_pa effects) vs. Genres")+
  geom_hline(yintercept=0, color="black")+
  labs(size="Movies (n)")+
  theme(plot.title = element_text(hjust=0.5))
p8

pdf("figs/genre-any_vs_residual.pdf")
p8
dev.off()

saveRDS(g, "coeff/genre-g.rds")

# As can be seen from the last plot, Horror or Action movies associate with negative residuals, implying overestimation by our current model, while Documentaries, War or Film-Noir movies have the most positive rating residuals, implying underestimation by our current model. It can also be seen that the ranking of the categories is quite well preserved when comparing to ranking of genres when we filtered for movies that had only one genre assigned to them. Therefore, it is reasonable to assume that using combinations of genres to deduct the influence of any single genre is reliable enough.

# Now we will incorporate the genre effects into our model.

# Pull the genre estimates and name the vector values with their corresponding genres
genre_values <- g$g
names(genre_values) <- g$genre
# Pull the genre category sizes into a separate vector and equally name them.
genre_sizes <- g$n
names(genre_sizes) <- g$genre

# Then we define a function to calculate the averages of the g_k values, i.e. since many movies have multiple genres assigned to them, we will average the effects of these genres per movie. We will also see if regularization has any benefit. We will perform the calculation on the test set.
# Our updated model is thus the following:
# $R_i,j = µ + m_i + u_j + rate_pa_i + average(g_k)_i + €_i,j$, with R_i,j being the expected rating of user j for movie i , µ the global mean, m_i the movie effect for movie i, u_j the user effect for user j, rate_pa_i the rating frequency effect for movie i, average(g_k)_i the average if genre effects g_k for movie i, €_i,j the residual for movie i and user j.

# Apply the function to each string in the avg_mov_rating genre vector and store the results in a matrix with l in rows and the averages of regularized g_k for each movie in columns
l <- seq(0,1000,100)
mov_genre_means <- sapply(avg_mov_rating$genre, function(s) {
  sapply(l, function(l){
    categories <- str_split(s, "\\|")[[1]]
    # Use the genre_values vector to look up the g_k values for each category.
    values <- genre_values[categories]
    # Use the genre_sizes vector to look up the sample sizes of each category.
    sizes <- genre_sizes[categories]
    # Regularize each genre contribution
    g_k_reg <- values*sizes/(sizes+l)
    # Average the values
    sum(g_k_reg, na.rm = TRUE)/length(g_k_reg)
  })
})

# We transpose the mov_genre_means matrix and assign column names (filled by the tuning parameter l)
mov_genre_means <- t(mov_genre_means)
colnames(mov_genre_means) <- l

# We have to loop through every column of mov_genre_means and obtain an RMSE on the test set.

genre_regul <- sapply(l, function(i) {
  # Add a g_i (averaged genres residual per movie i) column to the avg_mov_rating data set and remove the names of the vector
  avg_mov_rating <- avg_mov_rating %>% mutate(g_i=mov_genre_means[,colnames(mov_genre_means)==i])
  # add the g_i column to test data sets
  test <- avg_mov_rating %>% select(movieId, g_i)  %>% right_join(test, by="movieId", multiple="all")
  # Calculate the RMSE on the test set.
  mov_user_ratepa_genre_model_rmse <- RMSE(test$rating, (global_mean + test$g_i + test$m_i + test$u_j + test$rate_pa_i))
  mov_user_ratepa_genre_model_rmse
  #Does genre improve our fit over the movie/user/rating_pa model?
})

plot(l, genre_regul)
min(genre_regul)
l[which.min(genre_regul)]
 
# The minimal RMSE is obtained using l = 400 ! Let's use the l = 400 column of the mov_genre_means matrix as our final g_i estimate and add it to train and test sets.

avg_mov_rating <- avg_mov_rating %>% mutate(g_i=unname(mov_genre_means[,colnames(mov_genre_means)==400]))
# add the g_i column to test data set
test <- avg_mov_rating %>% select(movieId, g_i)  %>% right_join(test, by="movieId", multiple="all")
# Calculate the RMSE on the test set.
mov_user_ratepa_genre_model_rmse <- RMSE(test$rating, (global_mean + test$g_i + test$m_i + test$u_j + test$rate_pa_i))
rating_pa_user_movie_rmse - mov_user_ratepa_genre_model_rmse
paste0("RMSE with movie, user, movie rating frequency and genre effects: ",signif(mov_user_ratepa_genre_model_rmse, digits=7))
# We achieved a reduction in RMSE by 0.00012. Now it is 0.8650711.

# We add the g_i effect to the train data set as well.
train <- avg_mov_rating %>% select(movieId, g_i)  %>% right_join(train, by="movieId", multiple="all") 

rm(g, genre_list, gs, m_i, med_genre_rating, mov_genre_means, p7, p8, stats, genre_crowded, genre_regul, genre_sizes, genre_values, unique_genres)


############### STEP 5: YEAR effect ###############

# Next, we want to check if there is any correlation between the year of release and the remaining average residuals per movie. It might be that old movies were particularly entertaining or there was a peak in quality at some point, or maybe modern movies are the pinnacle of cinematography.

# Let's first plot the residuals against year of release.
p9 <- train %>% group_by(movieId) %>% 
  summarize(year = year[1], avg_residual = mean(rating - global_mean - m_i -u_j - rate_pa_i - g_i)) %>%
  ggplot(aes(year, avg_residual)) +
  geom_jitter(alpha=0.2, size=0.5)+
  geom_smooth(method="loess", span=0.1, method.args=list(degree=1), color="red")+
  geom_hline(yintercept=0, color="blue")+
  ylab("Average rating residual per movie ") +
  xlab("Year of release")+
  ggtitle("Residuals (post movie, user, rate_pa & genre effects) vs. Year")+
  theme(plot.title = element_text(hjust=0.5))
p9

pdf("figs/year_vs_residual.pdf")
p9
dev.off()
# We see, that there is some time-dependency, with movies createdbefore the 1940s having slightly positive residuals. In addition there are troughs around 1980 and 1995, resulting in slightly negative residuals. It might be worth to account for this effects.

# Our updated model is therefore:
# $R_i,j = µ + m_i + u_j + rate_pa_i + average(g_k)_i + yr_i + €_i,j$, with R_i,j being the expected rating of user j for movie i , µ the global mean, m_i the movie effect for movie i, u_j the user effect for user j, rate_pa_i the rating frequency effect for movie i, average(g_k)_i the average of genre effects g_k for movie i, yr_i the year of release effect for movie i, €_i,j the residual for movie i and user j.

# Let's obtain the number of movies per year of release to be able to use regularization later.
movies_pa <- train %>% group_by(movieId) %>% 
  summarize(year = year[1])%>% group_by(year) %>% summarize(n_y = n())

# Let's obtain the per year estimates with the loess function and optimize span, using RMSE on the test set as an indicator of how the span affects our estimates.
movie_yr <- train %>% group_by(movieId) %>% summarize(year=year[1]) # range of values of when our movies were released
avg_mov_rating <- avg_mov_rating %>% mutate(year = movie_yr$year) # adding the year values to the movie averages object

span <- seq(0.02, 0.3, 0.02)
rmses_age_loess <- sapply(span, function(s){
  
  data <- train %>% group_by(movieId) %>% 
    summarize(year = year[1], avg_residual = mean(rating - global_mean - m_i - u_j - rate_pa_i - g_i))
  loess_age <- data %>% loess(avg_residual ~ year, data=., span=s, degree=1)
  
  r_hat <- predict(loess_age, newdata=test)
  
  RMSE(test$rating, (r_hat + global_mean + test$m_i +test$u_j + test$g_i + test$rate_pa_i))
}
)
plot(span, rmses_age_loess)
min(rmses_age_loess)
span[which.min(rmses_age_loess)]

# We will use a span of 0.16 although the minimum is at 0.24. 0.16 seems to be more reasonable as we have very localized deviations from 0.
loess_age <- train %>% group_by(movieId) %>% 
  summarize(year = year[1], avg_residual = mean(rating - global_mean - m_i - u_j - rate_pa_i - g_i)) %>% 
  loess(avg_residual ~ year, ., span=0.16, degree=1)

# Let's obtain the per year residual effects yr_i for movie i
yr_i <- predict(loess_age, min(train$year):max(train$year))

# Let's now construct a function that pulls the yr_i effects and adds them to our current effect estimates and try to regularize the model with the "movies released per year" n_y parameter (via a movies_pa_vec vector with per year movie batch sizes).

names(yr_i) <- movies_pa$year #naming the yr_i vector with the release years
movies_pa_vec <- movies_pa$n_y # generating movies per year vector
names(movies_pa_vec) <- movies_pa$year #naming the movies_pa_vec vector with the release years

# Then we define a function to assign a year effect to each  year in test$year string vector and regularize.
l <- seq(0,1000, 100)
year_reg_model_rmses <- sapply(l, function(l){
  
  # Use the yr_i vector to look up the loess-estimated year effect values for each year and look up the movies_pa vector to estimate yearly movie cohort sizes.
  values <- yr_i[as.character(avg_mov_rating$year)]
  sizes <- movies_pa_vec[as.character(avg_mov_rating$year)]
  values <- unname(values)
  sizes <- unname(sizes)
  #Add a yr_i_reg column to the test data set, and regularize with the tuning parameter l
  yr_i_reg <- avg_mov_rating %>% mutate(yr_i_reg = values*sizes/(sizes+l))
  test <- yr_i_reg %>% select(movieId, yr_i_reg) %>% right_join(test, by="movieId", multiple="all")
  # Calculate the RMSE on the test set.
  year_reg_genre_rmse <- RMSE(test$rating, (global_mean + test$yr_i_reg + test$g_i + test$rate_pa_i + test$m_i + test$u_j))
}
)
year_reg_model_rmses
plot(l, year_reg_model_rmses)
min(year_reg_model_rmses)
l[which.min(year_reg_model_rmses)]

# l of 300 seems optimal. Let's add a yr_i column for l = 300 for train and test data sets.
values <- yr_i[as.character(avg_mov_rating$year)]
sizes <- movies_pa_vec[as.character(avg_mov_rating$year)]
values <- unname(values)
sizes <- unname(sizes) 
#Add a yr_i column to the movie averages object while regularizing with the tuning parameter l = 300. Then join to train and test data sets.
yr_i <- avg_mov_rating %>% mutate(yr_i = values*sizes/(sizes+300))
train <- yr_i %>% select(movieId, yr_i) %>% right_join(train, by="movieId",multiple="all")
test <- yr_i %>% select(movieId, yr_i) %>% right_join(test, by="movieId", multiple="all")

# Now we calculate the new RMSE for the movie/user/rat_pa/genre/year model for the test set. Is there any improvement over the previous model?
mov_user_ratepa_genre_year_model_rmse <- RMSE(test$rating, (global_mean + test$g_i + test$m_i + test$u_j + test$rate_pa_i + test$yr_i))
paste0("RMSE with movie, user, movie rating frequency, genre and year effects: ",signif(mov_user_ratepa_genre_year_model_rmse, digits=7))
mov_user_ratepa_genre_model_rmse - mov_user_ratepa_genre_year_model_rmse 

# The year factor does barely improve the model. The improvement was by 0.000066. Now the RMSE is at 0.8650.
rm(loess_age, movie_yr, movies_pa, p9, movies_pa_vec, rmses_age_loess, sizes, values, year_reg_model_rmses)

### STEP 6: REVIEW DATE ###

# Finally, we examine if the time of review affected the rating given. One possibility would be that over time users became more critically of movies and overall ratings declined or that sentiment trend was influenced in some other way. We will round to weeks to limit the number of time points to a reasonable amount.
p10 <- train %>% mutate(residual = (rating - global_mean - m_i - u_j - g_i - rate_pa_i - yr_i), date= round_date(rating_date, unit="week")) %>% group_by(date) %>% summarize(avg_residual=mean(residual), n=n()) %>%
  ggplot(aes(date, avg_residual))+
  geom_point(aes(size=n), alpha=0.3)+
  geom_smooth(method="loess", span=0.05, se=FALSE)+
  geom_hline(yintercept=0, alpha=0.7)+
  ylab("Average rating residual per rating week ") +
  xlab("Rating date (by weeks)")+
  labs(size="Ratings (n)")+
  ggtitle("Residuals (post movie, user, rate_pa, genre & year effects) vs. Date of rating")+
  theme(plot.title = element_text(hjust=0.5))
p10

pdf("figs/review-date_vs_residuals.pdf", width=8, heigh=6)
p10
dev.off()
# There seems to be a downtrend in rating residuals between the end of 90s and early 2000s with a small spike around 2004. It seems as if in the 90s the ratings were more positive overall, maybe due to enthusiasm about the beginnings of the internet.

# We are updating our model by this factor:
# $R_i,j = µ + m_i + u_j + rate_pa_i + average(g_k)_i + yr_i + revdate_w + €_i,j$, with R_i,j being the expected rating of user j for movie i , µ the global mean, m_i the movie effect for movie i, u_j the user effect for user j, rate_pa_i the rating frequency effect for movie i, average(g_k)_i the average of genre effects g_k for movie i, yr_i the year of release effect for movie i, revdate_w the effect of the rating time for week w, €_i,j the residual for movie i and user j.

# We will model the date of review  trend with the loess algorithm and first optimize the span by examiing RMSE changes when applying the review data prediction on the test set.

temp <- train %>% mutate(residual = (rating - global_mean - m_i - u_j - g_i - rate_pa_i - yr_i), date= round_date(rating_date, unit="week")) %>% group_by(date) %>% summarize(avg_residual=mean(residual), timestamp=timestamp[1], n=n())

span<- seq(0.025, 0.25, 0.025)
rmses_reviewdate_loess <- sapply(span, function(x){
  
  loess_reviewdate <-  loess(avg_residual ~ timestamp , data=temp, span=x, degree=1)
  
  test_temp <- test %>% mutate(residual = (rating - global_mean - m_i - u_j - g_i - rate_pa_i - yr_i), date= round_date(rating_date, unit="week")) %>% group_by(date) %>% summarize(avg_residual=mean(residual), timestamp=timestamp[1])
    
  r_hat <- predict(loess_reviewdate, newdata=test_temp)
  
  ind <- which(is.na(r_hat))  # there is a week in the test set that was not accounted for in the training set, we need to remove it, as it won't be estimated by the loess function.
  RMSE(test_temp$avg_residual[-ind], r_hat[-ind])
}
)
plot(span, rmses_reviewdate_loess)
span[which.min(rmses_reviewdate_loess)]

# We will use a span of 0.025 to train the loess model.
loess_reviewdate <- train %>% mutate(residual = (rating - global_mean - m_i - u_j - g_i - rate_pa_i - yr_i), date= round_date(rating_date, unit="week")) %>% group_by(date) %>% summarize(avg_residual=mean(residual), timestamp=timestamp[1]) %>% 
  loess(avg_residual ~ timestamp, ., span=0.025, degree=1)

# Let's obtain the review date-dependent effect revdate_w for week w over a range of observed values.
revdate_w <- predict(loess_reviewdate, newdata=temp)

# Let's now pull the revdate_w effects for each entry in the training data set and construct a function that regularizes the effects by a tuning parameter and then computes RMSE on the test data set.

# We give the revdate_w vector names of the dateweek and generate a vector rev_pw_sizes with the weekly review batch sizes
names(revdate_w) <- temp$date
rev_pw_sizes <- temp$n
names(rev_pw_sizes) <- temp$date

# Add the date, rounded by week, to train and test data sets.
train <- train %>% mutate(dateweek = round_date(rating_date, unit="week"))
test <- test %>% mutate(dateweek = round_date(rating_date, unit="week"))

  # Use the revdate_w vector to match the values to each entry in the train data, by dateweek, and do the same using the rev_pw_sizes vector to determine the weekly review number for that week for regularization purposes
  values <- revdate_w[as.character(train$dateweek)]
  sizes <- rev_pw_sizes[as.character(train$dateweek)]
  values <- unname(values)
  sizes <- unname(sizes)
  
   #Tuning parameter
  l <- seq(0, 1000, 100)
  
  # Then we define a function to assign a revdate_w effect to each stratum of rev_pw in the train$dateweek vector and regularize with tuning parameter l.
  revdate_reg_model_rmses <- sapply(l, function(l){
  #Add a revdate_w effect column to the train data set, and regularize it with the tuning parameter l
    train_by_date <- train %>% mutate(revdate_w = values*sizes/(sizes+l)) %>% group_by(dateweek) %>% 
      summarize(revdate_w=revdate_w[1]) %>% select(dateweek, revdate_w)
  # Add the revdate_w effect to the rating prediction in the test set
  temp <- test %>% left_join(train_by_date, by="dateweek", multiple="all") %>% mutate(r_hat = global_mean + m_i + u_j + g_i + rate_pa_i + yr_i + revdate_w) %>% select(rating, r_hat)
  # Calculate the RMSE.
  revdate_reg_rmse <- RMSE(temp$rating, temp$r_hat)
  revdate_reg_rmse
}
)
plot(l, revdate_reg_model_rmses)
l[which.min(revdate_reg_model_rmses)]

# Regularization has no benefit here. Therefore, let's just add a revdate_w column to the train and test data sets.
values <- revdate_w[as.character(train$dateweek)]
values <- unname(values)

train <- train %>% mutate(revdate_w = values) 

values <- revdate_w[as.character(test$dateweek)]
values <- unname(values)

test <- test %>% mutate(revdate_w = values) 

# Now we calculate the new RMSE for the movie/user/rat_pa/genre/year/revdate model for the test set. Is there any improvement over the previous model?
revdate_year_genre_mov_user_ratepa_model_rmse <- RMSE(test$rating, (global_mean + test$g_i + test$m_i + test$u_j + test$rate_pa_i + test$yr_i +test$revdate_w))
paste0("RMSE with movie, user, movie rating frequency, genre, year and review date effects: ",signif(revdate_year_genre_mov_user_ratepa_model_rmse, digits=7))
mov_user_ratepa_genre_year_model_rmse - revdate_year_genre_mov_user_ratepa_model_rmse 

# We improved the loss of our prediction by additional 0.00011 units. Now the RMSE is 0.86489

############### STEP 7: USER-GENRE effects ###############
#Finally, we will consider if users have preferences for certain genres. We obtain the residuals and calculate the average residuals per user and per genre.

#creating a genre list from the light object avg_mov_rating and again filter out the unique genres
genre_list <- str_split(avg_mov_rating$genre, "\\|")
unique_genres <- sort(unique(unlist(genre_list)))
unique_genres
which(unique_genres == "IMAX")
unique_genres <- unique_genres[-c(1, which(unique_genres == "IMAX"))] # removing "(no genres listed)" and "IMAX"
unique_genres

# creating a matrix matching rows of the train set with all applicable genre assignments for each movie split into separate columns
genre_sep <- str_split(train$genres, "\\|", simplify = TRUE)
# add column names to each of the 8 resulting columns
colnames(genre_sep) <- c(paste0("genre",seq(1:ncol(genre_sep))))

# create an empty (filled with NAs) matrix with as many rows as the train set and as many columns as there are unique genres
genre_user_matrix <- matrix(NA, nrow = length(train$genres), ncol = length(unique_genres), dimnames = list(train$userId, unique_genres))

# Calculate current residual errors
residuals <- train %>% mutate(residual = rating - global_mean - m_i - u_j - rate_pa_i - g_i - yr_i - revdate_w) %>% pull(residual)

length(residuals)==nrow(genre_user_matrix)

# We create a loop to insert a rating residual, corresponding to every genre category that a movieId rated by a userId fulfills, into the genre_user_matrix, and leave "NA" for genres that a rated movie does not fit in. ###COMPUTATIONALLY INTENSIVE ###

for (i in 1:nrow(genre_sep)) {
  for (j in 1:length(unique_genres)) {
    if (unique_genres[j] %in% genre_sep[i,]) {
      genre_user_matrix[i, j] <- residuals[i]
    }
  }
}

dim(genre_user_matrix) # check for correct dimensions 
genre_user <- as.data.frame(userId=train$userId, genre_user_matrix[,1:18]) #create a data frame with userIds matching the columns of the newly generated matrix
genre_user <- genre_user %>% mutate(userId= train$userId) #assert userId column

# calculate the mean residuals for each user and genre, add a "n" column reflecting the amount of reviews each user has submitted. Resulting object stores the g_jk effects for user j and genre k.
genre_user_means <- genre_user %>% group_by(userId) %>% summarize(across(everything(), ~mean(., na.rm=TRUE)), n=n()) 
genre_user_means <- replace(genre_user_means, is.na(genre_user_means), 0) # replace NA values with 0

#visualize residuals across genres for random picked 5 users that rate very frequently 
temp <- genre_user_means %>% filter(userId %in% c("58357","42791", "63134", "30723", "56707")) %>% pivot_longer(cols=c(-userId, -n),names_to="genre", values_to="residual", ) 
temp %>% ggplot(aes(x=genre, y=residual, group=userId, color=factor(userId)))+
  geom_point(alpha=0.5, size=1)+
  geom_line()+
  theme(axis.text.x = element_text(angle = 90, vjust=0.5, hjust =1.0))+
  ylab("Average residual per user")+
  xlab("Genre")+
  ggtitle("Residuals per user vs. genre ")+
  geom_hline(yintercept=0, color="black")+
  theme(plot.title = element_text(hjust=0.5, size=10))
# It is quite evident that users have differential likings and antipathies. 

# Purpose of following code: extract the separate movie genres from each user rating in the test set, match it with the userId from the genre_user_means object and pull and average the corresponding genre residuals to compute a g_ij effect value. Try regularization using the "n" column from genre_user_means.

genre_sep_test <- str_split(test$genres, "\\|", simplify = TRUE)
colnames(genre_sep_test) <- c(paste0("genre",seq(1:ncol(genre_sep_test))))

genre_sep_test <- as.data.frame(genre_sep_test)

# Create a function that counts how many genres are assigned to each row
count_non_empty <- function(row) {
  sum(!is.na(row) & row != "")
}

# Apply the function to each row in separated-genre data frame
genre_sep_test$genre_count <- apply(genre_sep_test, MARGIN=1, count_non_empty)
all(genre_sep_test$userId == test$userId) # verify if data frames align by userId

genre_sep_test <- genre_sep_test %>% mutate(userId=test$userId, movieId=test$movieId, g_ij=0, rowId=rownames(.)) # make sure that all information relevant to the rating, most importantly, the initial row index, are kept, so data can be reunited with the test data set. Create a g_ij column, storing the genre effect for user j and movie i, for now equal to 0

merge <- merge(genre_sep_test, genre_user_means, by="userId") # join genre_sep_test data frame with the object containing all user preferences. Now each test set row contains the values for each user's genre preference effects,

# function to calculate sum of genre effect values for each row matching the genre combination of the rated movie
calc_sum <- function(row) {
  genres <- row[2:9] #create a vector containing the genres of the rated movie for a row
  values <- row[14:31] # create vector of the user's genre preference effect values
  sum_values <- sapply(1:length(genres), function(i) {
    values[which(names(values) == as.character(genres[i]))]
  }) # run a function that finds which genres are being rated in the row and pulls the corresponding values from the values vector
  sum_values <- unlist(sum_values, use.names = FALSE) # sum up effects
  sum <- sum(as.numeric(sum_values)) # make sure the sum is numeric
  sum
}

# apply the function to each row of "merge" and store the result in a g_ij_sum vector
g_ij_sum <- apply(merge, 1, calc_sum)

# divide g_ij_sum by the amount of genres assigned to the movie and store it in a g_ij column
merge <- merge %>% mutate(g_ij = g_ij_sum / genre_count)
# modify the test data set by calculating the current prediction and inserting a column with the row indices
test <- test %>% mutate(r_hat = global_mean + m_i +u_j + rate_pa_i + g_i + yr_i + revdate_w, rowId= rownames(.))

# regularize by user activity (stored in column "n", count for how many ratings each user has submitted), and test RMSE
l <- seq(0,100,10)
userxgenre_rmses <- sapply(l, function(l){
  test <- merge %>% mutate(g_ij_reg = g_ij*n/(n+l)) %>% select(g_ij_reg, rowId) %>% right_join(test, by="rowId")
  r_hat <- test %>% mutate(r_hat= r_hat + g_ij_reg) %>% pull(r_hat)
  RMSE(test$rating, r_hat)
  })
plot(l, userxgenre_rmses)
min(userxgenre_rmses)
l[which.min(userxgenre_rmses)]
# Tuning parameter of 20 is optimal.

# now regularize and then join the g_ij column from merge to the test data set. It is important to join by "rowId" since neither the userIds, nor movieIds are aligned any more between "merge" and "test"
test <- merge %>% mutate(g_ij = g_ij*n/(n+20)) %>% select(g_ij, rowId) %>% right_join(test, by="rowId")
userxgenre_rmse <- RMSE(test$rating, test$r_hat+test$g_ij) # RMSE of updated model on the test set
userxgenre_rmse
paste0("RMSE with movie, user, movie rating frequency, genre, year, review date and user/genre interaction effects: ",signif(userxgenre_rmse, digits=7))
test <- test %>% select(-r_hat) # remove temporary r_hat column

# We repeat the very same procedure to compute the g_ij values for the train set

genre_sep <- as.data.frame(genre_sep)

# Apply the function that counts how many genres are assigned to each row
genre_sep$genre_count <- apply(genre_sep, MARGIN=1, count_non_empty)
all(genre_sep$userId == train$userId) # verify that both data frames are aligned by userId

genre_sep <- genre_sep %>% mutate(userId=train$userId, movieId=train$movieId, g_ij=0, rowId=rownames(.)) # make sure the data frame contains all identifying information, including row index

merge_train <- merge(genre_sep, genre_user_means, by="userId") #fetch the user preferences and adjoin them to the genre_sep data frame

# apply the calc_sum function to each row of the train set derived merge_train object and store the result in a g_ij_sum vector. COMPUTATION INTENSIVE!!! ###
g_ij_sum <- apply(merge_train, 1, calc_sum)

# divide g_ij_sum by the amount of genres assigned to the movie and regularize to generate g_ij column
merge_train <- merge_train %>% mutate(g_ij = g_ij_sum / genre_count) %>% mutate(g_ij = g_ij*n/(n+20)) 

train <- train %>% mutate(rowId= rownames(.)) # add current prediction and make a column with row indices

train <- merge_train %>% select(g_ij, rowId) %>% right_join(train, by="rowId") # include g_ij into the train data.frame

rm(loess_reviewdate, p10, temp, revdate_reg_model_rmses, rmses_reviewdate_loess, sizes, values)

############### STEP 8: PREDICTED vs. TRUE rating assessment ###############

# Lastly, we should look at the distribution of our finalized predictions against the true ratings to evaluate the utility of our current model.

# Add the prediction from our final model to the train and test data
train <- train %>% mutate(final_r_hat = global_mean + m_i + u_j + g_i + rate_pa_i + yr_i + revdate_w + g_ij)
test <- test %>% mutate(final_r_hat = global_mean + m_i + u_j + g_i + rate_pa_i + yr_i + revdate_w + g_ij)

# Save the finalized train and data sets
saveRDS(train, "data/train_model.rds")
saveRDS(test, "data/test_model.rds")
train <- readRDS("data/train_model.rds")
test <- readRDS("data/test_model.rds")

# Histogram of true ratings
hist(test$rating)
# Histogram of predicted ratings
hist(test$final_r_hat)
# Predicted rating against true ratings as boxplots. Blue horizontal line indicates the overall average of movie rating averages, which we used as our initial naive prediction.

p11 <- test %>% ggplot(aes(factor(rating), final_r_hat)) +
  geom_boxplot(outlier.size=0.25) +
  geom_hline(yintercept=c(0.5,5), lty = 2) + 
  scale_y_continuous(limits=c(-0.7, 6.4), breaks=c(seq(-0.5,6.0,0.5)))+
  geom_hline(yintercept=global_mean, color = "blue")+
  xlab("True rating")+
  ylab("Predicted rating")+
  ggtitle("Performance of final model")+
  theme(plot.title = element_text(hjust=0.5))
p11

pdf("figs/predicted_vs_true.pdf")
p11
dev.off()

# Our predictions per true rating stratum generally show a good ascending trend. The correlation is quite high.
paste0("Correlation between predicted and true ratings: ",signif(cor(test$rating, test$final_r_hat), digits=4))

# We can see that some of our predicted ratings extend over the natural range of the ratings. We should consider capping our predictions. While it is reasonable to cap them to 0.5 as minimum and 5.0 as maximum, it is not immediately clear if this would be the optimum, since our failed estimates extend over a large range of true ratings. Therefore we will build a function to find the optimum. The floor should be somewhere below the global mean and the ceiling above the global mean, however.

############### STEP 9: DATA CLIPPING OPTIMIZATION ###############

floor <- seq(0.5, 3.2, 0.1)
ceiling <- seq(3.3, 5.0, 0.1)

limits_optim <- sapply(ceiling, function(ceiling){
  sapply(floor, function(floor){
    train <- train %>% mutate(final_r_hat = pmax(pmin(final_r_hat, ceiling), floor))
    RMSE(train$rating, train$final_r_hat)
  })
})
colnames(limits_optim) <- ceiling
rownames(limits_optim) <- as.character(floor)

#look up where the minimum RMSE is in the limits_optim matrix and to which floor/ceiling values it corresponds
ind <- which(limits_optim == min(limits_optim), arr.ind = TRUE)
ind
floor[ind[,1]]
ceiling[ind[,2]]
floor <- floor[ind[,1]]
ceiling <- ceiling[ind[,2]]

#Thus, it is optimal to cap our predicted rating values at 0.8 for the floor and 4.8 for the ceiling. This will decrease the impact of our misses and make our predictions more conservative.

# By how much do we improve our loss when applying these caps to the test data?
final_cap_RMSE <- RMSE(test$rating,  pmax(pmin(test$final_r_hat, ceiling), floor)) 
paste0("RMSE with movie, user, rating frequency, genre, year and review date effects and prediction capping: ",signif(final_cap_RMSE, digits=7))
userxgenre_rmse - final_cap_RMSE

# Our RMSE is reduced by 0.00048 points and the value is now at 0.8499.

#How many of the predicted rating values are above 4.8 or below 0.8?

mean(train$final_r_hat>ceiling) + mean(train$final_r_hat<floor)

mean(test$final_r_hat>ceiling) + mean(test$final_r_hat<floor)

# About 0.9 % of predicted ratings are affected by our capping method which seems reasonable.

# Final plot with capped predictions
p12 <- test %>% ggplot(aes(factor(rating), pmax(pmin(final_r_hat, ceiling), floor))) +
  geom_boxplot(outlier.size=0.25) +
  geom_hline(yintercept=c(0.5,5), lty = 2) + 
  scale_y_continuous(limits=c(-0.7, 6.0), breaks=c(seq(-0.5,6.0,0.5)))+
  geom_hline(yintercept=global_mean, color = "blue")+
  xlab("True rating")+
  ylab("Capped predicted rating")+
  ggtitle("Performance of final model (capped)")+
  theme(plot.title = element_text(hjust=0.5))
grid.arrange(p11, p12, nrow=1)

pdf("figs/cap-predicted_vs_true.pdf", width=10, height=6)
grid.arrange(p11, p12, nrow=1)
dev.off()

paste0("Correlation between capped predicted and true ratings: ",signif(cor(test$rating, pmax(pmin(test$final_r_hat, ceiling), floor)), digits=4))


# We compare each step of our incrementally developed model

rmses_compare <- data.frame(model = 
                              c("Average", "Movie", "Movie User", "Movie User Rating.pa", "Movie User Rating.pa Genre", "Movie User Rating.pa Genre Year", "Movie User Rating.pa Genre Year Revdate",  "Movie User Rating.pa Genre Year Revdate UserxGenre", "Movie User Rating.pa Genre Year Revdate UserxGenre CAP"), 
                            RMSE = signif(c(global_mean_rmse, movie_rmse, user_movie_rmse, rating_pa_user_movie_rmse, mov_user_ratepa_genre_model_rmse, mov_user_ratepa_genre_year_model_rmse, revdate_year_genre_mov_user_ratepa_model_rmse, userxgenre_rmse, final_cap_RMSE), digits=5),
                            Improvement = format(c(0,
global_mean_rmse - movie_rmse,
movie_rmse - user_movie_rmse,
user_movie_rmse - rating_pa_user_movie_rmse,
rating_pa_user_movie_rmse - mov_user_ratepa_genre_model_rmse,
mov_user_ratepa_genre_model_rmse - mov_user_ratepa_genre_year_model_rmse,
mov_user_ratepa_genre_year_model_rmse  - revdate_year_genre_mov_user_ratepa_model_rmse,
revdate_year_genre_mov_user_ratepa_model_rmse - userxgenre_rmse,
userxgenre_rmse - final_cap_RMSE), scientific=FALSE, digits=1))
rmses_compare

# Each additional considered effect improved the prediction.

# We are going to save the computed effects for later retrieval and sharing purposes.
effects <- train %>% group_by(movieId) %>% summarize(year = year[1], yr_i = yr_i[1], g_i = g_i[1], m_i = m_i[1], rate_pa_i = rate_pa_i[1])
effects <- effects %>% mutate (avg_rating = avg_mov_rating$avg_rating, genre = avg_mov_rating$genre, n_reviews = avg_mov_rating$n)

rm(ind, limits_optim, p11, l, span, yr_i, p12, genre_list, genre_sep, genre_sep_test, genre_user, genre_user_matrix, genres, merge, merge_train, p_values, wilcox_results, g_ij_sum, residuals, userxgenre_rmses, i, j)

saveRDS(u_j, "coeff/user_reg_effects.rds")
saveRDS(effects, "coeff/movie_attribute_effects.rds")
saveRDS(revdate_w, "coeff/review_week_effects.rds")
saveRDS(rev_pw_sizes, "coeff/review_week_sizes.rds")
saveRDS(genre_user_means, "coeff/genre_user_means.rds")

train_genre_user_effect <- train %>% select(rowId, userId, movieId, g_ij)
saveRDS(train_genre_user_effect, "coeff/train_genre_user_effect.rds")

test_genre_user_effect <- test %>% select(rowId, userId, movieId, g_ij)
saveRDS(test_genre_user_effect, "coeff/test_genre_user_effect.rds")

u_j <- readRDS("coeff/user_reg_effects.rds")
effects <- readRDS("coeff/movie_attribute_effects.rds")
revdate_w <- readRDS("coeff/review_week_effects.rds")
rev_pw_sizes <- readRDS("coeff/review_week_sizes.rds")
genre_user_means <- readRDS("coeff/genre_user_means.rds")

train_genre_user_effect <- readRDS("coeff/train_genre_user_effect.rds")
test_genre_user_effect <- readRDS("coeff/test_genre_user_effect.rds")

############### STEP 10: VALUATION OF MODEL PERFORMANCE ON FINAL HOLDOUT TEST ###############

### We will now test the model on the final holdout test ###

final_holdout_test <- readRDS("data/final_holdout_test.rds")

# add user effects
final_holdout_test <- final_holdout_test %>% left_join(u_j, by="userId")

# add all other effect
final_holdout_test <- effects %>% select(-genre, -n_reviews) %>% right_join(final_holdout_test, by="movieId", multiple="all") 
# create rating date and week columns
final_holdout_test <-final_holdout_test %>% mutate(rating_date = as_datetime(timestamp), dateweek = round_date(rating_date, unit="week"))

# pull revdate_w estimates for each entry in final_holdout_test and add them to the data set. Calculate final capped prediction.
values <- revdate_w[as.character(final_holdout_test$dateweek)]
values <- unname(values)
final_holdout_test <- final_holdout_test %>% mutate(revdate_w = values)

# Now we add the individual user's genre preferences. And make sure to generate a rowId column in final_holdout_test, to later rejoin.
final_holdout_test <- final_holdout_test %>% mutate(rowId = rownames(.))
genre_sep_final <- str_split(final_holdout_test$genres, "\\|", simplify = TRUE)
# add column names to each of the 8 resulting columns
colnames(genre_sep_final) <- c(paste0("genre",seq(1:ncol(genre_sep_final))))
genre_sep_final <- as.data.frame(genre_sep_final)

# Counts how many genres are assigned to each row
# Apply the function to each row
genre_sep_final$genre_count <- apply(genre_sep_final, MARGIN=1, count_non_empty)
all(genre_sep_final$userId == final_holdout_test$userId)

genre_sep_final <- genre_sep_final %>% mutate(userId=final_holdout_test$userId, movieId=final_holdout_test$movieId, g_ij=0, rowId=rownames(.))

merge_final <- merge(genre_sep_final, genre_user_means, by="userId")

# apply the calc_sum function to each row of the final holdout test set and store the result in a g_ij_sum vector. ###
g_ij_sum <- apply(merge_final, 1, calc_sum)

# divide g_ij_sum by the amount of genres assigned to the movie and regularize to generate g_ij column
merge_final <- merge_final %>% mutate(g_ij = g_ij_sum / genre_count) %>% mutate(g_ij = g_ij*n/(n+20)) 

final_holdout_test <- merge_final %>% select(g_ij, rowId) %>% right_join(final_holdout_test, by="rowId")

# save g_ij effect for final_holdout_test set in an object together with rowId, userId and movieId for later retrieval and joining with final_holdout_test
final_holdout_genre_user_effect <- final_holdout_test %>% select(rowId, userId, movieId, g_ij)
saveRDS(final_holdout_genre_user_effect, "coeff/final_holdout_genre_user_effect.rds")

final_holdout_test <- final_holdout_test %>% mutate(final_r_hat = pmax(pmin(global_mean + m_i + u_j + rate_pa_i + g_i + yr_i + revdate_w + g_ij, ceiling), floor)) 

evaluation <- RMSE(final_holdout_test$rating, final_holdout_test$final_r_hat)
paste0("RMSE when applying final model and capping to the final holdout test set: ", signif(evaluation, digits=7))

### RMSE on final holdout test is: 0.849893 ###

# removing all non-essential large objects
rm(avg_mov_rating, effects, final_holdout_genre_user_effect, genre_sep_final, genre_user_means, merge_final, test_genre_user_effect, train_genre_user_effect, u_j, g_ij_sum, values  )
