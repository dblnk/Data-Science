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

dl <- "ml-10M.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
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

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###################
# QUIZ
###################

# Q1

dim(edx)[1]
dim(edx)[2]

# Q2

sum(edx$rating=="0")
sum(edx$rating=="3")

# or
edx %>% filter(rating == 3) %>% tally()

# Q3
length(unique(edx$movieId))
# or, faster and more concise
n_distinct(edx$movieId)

# Q4
length(unique(edx$userId))
# or, faster and more concise
n_distinct(edx$userId)

# Q5
table(edx$genres) %>% knitr::kable()
genre_count <- tibble(n = table(edx$genres), genre = names(table(edx$genres)))
genre_count %>% filter(str_detect(genre, "Drama")) %>% summarize(sum_drama = sum(n))
genre_count %>% filter(str_detect(genre, "Comedy")) %>% summarize(sum_comedy = sum(n))
genre_count %>% filter(str_detect(genre, "Thriller")) %>% summarize(sum_thriller = sum(n))
genre_count %>% filter(str_detect(genre, "Romance")) %>% summarize(sum_romance = sum(n))

# provided answer # str_detect
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})
# interesting: # separate_rows, much slower!
#edx %>% separate_rows(genres, sep = "\\|") %>%
#  group_by(genres) %>%
#  summarize(count = n()) %>%
#  arrange(desc(count))

# Q6
most_rat <- edx %>% group_by(movieId) %>% tally() %>% arrange(-n) %>% top_n(1) %>% pull(movieId)
edx %>% filter(movieId == most_rat) %>% slice(1) %>% pull(title)

# provided answer
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Q7
edx %>% group_by(rating) %>% tally() %>% arrange(-n) %>% top_n(5)
# Q8
edx %>% group_by(rating) %>% tally() %>% arrange(-n) %>% top_n(5)
#provided answer in addition;
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()



setwd("~/R work/R course/Capstone/Movielens")
#We save the data sets in order to load them directly in the future.

saveRDS(edx, "edx.rds")
saveRDS(final_holdout_test, "final_holdout_test.rds")

edx <- readRDS("data/edx.rds")

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")

library(lubridate)
library(tidyverse)
library(caret)
library(dslabs)
library(broom)
library(purrr)
# We will undertake an agnostic approach, i.e. we have no idea what could influence the rating of a movie and we even don't know what a rating of, e.g., 1 versus a rating of 5 really means, i.e. which movies we want to recommend to a viewer or subscriber of the streaming or video rental service. 

# Let's first extract the year from the movie title (-> year), clean up the title by removing the year, create a rating date from the timestamp (-> rating_date), extract the year of the rating (-> rating_year) and calculate an interval between the year of the rating and the year of release (-> interval) to acknowledge the fact, that highly rated movies might associate with longer intervals as they can be viewed as those which stood the test of time.

options(digits=7)

edx %>% slice(1:100) %>% mutate(year = parse_number(str_extract(title, "\\(\\d{4}\\)$")), 
                               title = str_replace(title, "\\s\\(\\d{4}\\)$", ""),
                               rating_date = as_datetime(timestamp), 
                               rating_year = year(rating_date),
                              interval = rating_year - year) %>% str()

# Now we also want extract the single genres that the movies are associated with. Eventually, we want to look at those movies that have only one genre assigned, to disentangle the genre ambiguities and explore if singular genres have an influence on the rating outcome or if rating does not depend on the genre.

edx %>% slice(1:100) %>% filter(str_detect(genres, "^\\w+\\s*$"))




# First we split the edx data into train and test set.
set.seed(1998)
test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
train <- edx[-test_index, ]
temp <- edx[test_index, ]

# verify if all movieId and userId values are present in both train and test sets.
# Make sure userId and movieId in final hold-out test set are also in edx set
test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add removed rows from test set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

rm(test_index, temp, removed)

# Let's start to investigate visually if any particular feature is correlated with rating

### EFFECT OF THE GENRE ###

# Let's look at how the ratings are distributed across the unambiguous genre categories. We will average the ratings that each movie has received. And we will order the factor levels of genre by the ascending order of their medians.

avg_genre_rating <- train %>% filter(str_detect(genres, "^\\w+\\-?\\w+\\s*$")) %>%
  group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), genre = genres[1]) %>% group_by(genre) %>%
  summarize(avg_genre_rating = median(avg_rating)) %>% mutate(genre = reorder(genre, avg_genre_rating)) 

ds_theme_set()

train %>% filter(str_detect(genres, "^\\w+\\-?\\w+\\s*$")) %>%
  group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), genre = genres[1]) %>%
  mutate(genre = factor(genre, levels=levels(avg_genre_rating$genre))) %>%
  ggplot(aes(genre, avg_rating)) + 
  geom_boxplot(outlier.shape = NA) + 
  geom_jitter(alpha=0.2, size=1)+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# It is apparent from this plot that some genres, such as "Children" or "Fantasy" are filled with only a few movies, hence they are likely to be rarely the only genre attributes to a movie. Therefore, we will not be able to reliably estimate the effect of a singular genre for most genres. However, to make a general case we will have a look if there are any significant differences between categories that are sufficiently crowded (n>50), i.e. "Horror", "Action", Comedy", "Thriller", "Western", "Drama" and "Documentary".
genre_crowded <- train %>% filter(str_detect(genres, "^\\w+\\-?\\w+\\s*$")) %>%
  group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), genre = genres[1]) %>%
  mutate(genre = factor(genre, levels=levels(avg_genre_rating$genre))) %>%
  group_by(genre) %>% tally() %>% arrange(desc(n)) %>% filter(n>=50) %>%pull(genre)

stats <- train %>% filter(str_detect(genres, "^\\w+\\-?\\w+\\s*$")) %>%
  group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), genre = genres[1]) %>%
  mutate(genre = factor(genre, levels=levels(avg_genre_rating$genre))) %>% filter(genre %in% genre_crowded)

stats %>%
  ggplot(aes(genre, avg_rating)) + 
  geom_boxplot(outlier.shape = NA) + 
  geom_jitter(alpha=0.2, size=1)+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#First we check for normality

stats %>%
  group_by(genre) %>%
  summarize(test = list(shapiro.test(avg_rating)),
            pvalue = test %>% map_dbl("p.value"),
            normality = ifelse(pvalue > 0.05, "normal", "not normal"))
# For all but one category the assumption of normality is violated. Therefore we are using the non-parametric Wilcoxon rank-sum test.
pairwise.wilcox.test(stats$avg_rating, stats$genre,
                     p.adjust.method= "BH")
# As can be seen, all comparisons, have a significant p-value <0.05 after Benjamini-Hochberg multiple-testing correction. Assuming no confounders, we conclude that genre has a significant effect on movie ratings.

# To compute a coefficient for how much any genre influences the rating, we will filter for genres assignments to contain each of the available genres (except for "IMAX") separately and then group by movieId. I.e. if a movie is categorized as both "Fantasy" and "Action", it will contribute to the average ratings within both "Fantasy" and "Action" genres. We then subtract the average of all (averaged per movie) movie ratings in the train dataset (the global mean, giving equal weight to each movie) from the category average (of averages per movie) to obtain the genre coefficient g.

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
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

saveRDS(g, "R course/Capstone/Movielens/data/genre-g.rds")
# As can be seen from the last graph, horror movies have the most negative influence on a movie rating, while documentaries and war movies have the most favorable effect on movie ratings. It can also be seen that the ranking of the categories perfectly matches the ranking of genres when we filtered for movies that had only one movie assigned to them. Therefore, it is safe to assume that using combinations of genres to deduct the influence of any single genre is reliable enough.

### RELEASE YEAR ###

# Next we want to check if there is any correlation between the year of release on the average per movie movie ratings. It might be that old movies were particularly entertaining or there was a peak in quality at some point, or maybe modern movies are the pinnacle of cinematography.

train_years <- train %>% mutate(year = parse_number(str_extract(title, "\\(\\d{4}\\)$")), 
                                title = str_replace(title, "\\s\\(\\d{4}\\)$", ""),
                                rating_date = as_datetime(timestamp), 
                                rating_year = year(rating_date),
                                interval = rating_year - year) 
p1 <- train_years %>% group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), year = year[1]) %>%
  ggplot(aes(year, avg_rating)) +
  geom_jitter(alpha=0.2, size=0.5)+
  geom_smooth(method="loess", span=0.3, method.args=list(degree=1))+
  ylab("Average movie rating")+
  xlab("Release year")
p1
# We acknowledge that are only a few movies released before 1930s. However, the general trend is that old movies have higher average ratings. Additionally, there seems to be a trough in movie ratings in the 1980s. Finally, there have been many more movies released since the 1990s including the worst rated ones. It seems reasonable to consider an effect of the release year. However, we have to keep in mind that old movies might be sought after by movie enthusiasts and be among the subset of movies that have endured the test of time and are still being watched. Other old movies might be of much lower quality. Recommending an old movie just because it is old is seems therefore not reasonable. Nevertheless, we can only recommend movies that are contained in the train set, therefore we might consider this "surivval" effect as a legit parameter.

# Another way to look at the time effect is to observe the interval between release date and movie rating. We might expect, that longer intervals associate with better ratings, since such movies have survived oblivion.
p2 <- train_years %>% group_by(movieId) %>% 
  summarize(avg_rating = mean(rating), interval = interval[1]) %>%
  ggplot(aes(interval, avg_rating)) +
  geom_jitter(alpha=0.2, size=0.5)+
  geom_smooth(method="loess", span=0.3, method.args=list(degree=1))+
  ylab("Average movie rating")+
  xlab("Interval (years) between rating and release year")
p2
library(gridExtra)
grid.arrange(p1, p2, nrow=1)

## TO DO, extract the avg_movie rating estimate from the loess function.

#Very similarly, we observe a time effect. It seems that the movies from the 1990s-2000s are condensed into an interval of 5 years, likely due to most reviews of these movies being released shortly after release date and the upcoming of the internet in the 1990s. There is a trough in ratings for an interval of 10 years and an increased approval rating the further back the release date is. Eventually, we want to stick with the release year predictor, since we want to avoid the condensation effects observed in the avg_rating vs. interval plot.


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


#Let's check for correlation, but filtering out small values first

train_ratings_pa <- train_years %>% group_by(movieId) %>% 
  summarize(ratings_pa = n()/(2009-year[1])/dim(train)[1]*10^6, avg_rating=mean(rating)) 

train_ratings_pa %>% filter(ratings_pa > quantile(ratings_pa, 0.10)) %>% summarize(cor = (cor(sqrt(ratings_pa), avg_rating, method="spearman")))

#The pearson correlation coefficient is r = 0.166

# Let's construct intervals of the rating per year and the associated average rating minus the global mean
range(sqrt(train_ratings_pa$ratings_pa))

train_ratings_pa %>% mutate(strata = cut(sqrt(train_ratings_pa$ratings_pa), breaks = seq(0, 16.2, by = 1.62))) %>%
  group_by(strata) %>% 
  summarize(mean_rating = mean(avg_rating - global_mean), 
            lower = mean_rating - 1.96*sd(avg_rating)/sqrt(n()), 
            upper = mean_rating + 1.96*sd(avg_rating)/sqrt(n()), n= n() ) %>% 
  ggplot(aes(strata, mean_rating, ymin=lower, ymax=upper))+ 
  geom_point()+
  geom_errorbar()+
  ylab("Average movie rating")+
  xlab("Strata of Sqrt(Norm. ratings per year)")

# This plot shows a clear trend between rating rate and rating score. The less reviews a movie gets, the noisier and wide-spread the rating score distribution is. The trend in this plot is very clear with frequently reviewed movies receiving higher scores. Ratings per year might allow for regularization of other predictors, penalizing rarely viewed and rated, therefore likely obscure, movies.

# Let's save the average ratings per stratum as separate data.frames containing the "ratings per year", quasi, bonus, coefficient.
pa <- train_ratings_pa %>% mutate(sqrt_pa_strata = cut(sqrt(train_ratings_pa$ratings_pa), breaks = seq(0, 16.2, by = 1.62))) %>%
  group_by(sqrt_pa_strata) %>% 
  summarize(mean_rating = mean(avg_rating - global_mean), 
            lower = mean_rating - 1.96*sd(avg_rating)/sqrt(n()), 
            upper = mean_rating + 1.96*sd(avg_rating)/sqrt(n()), n= n() ) %>% select(sqrt_pa_strata, mean_rating)

saveRDS(pa, "R course/Capstone/Movielens/data/ratings-pa.rds")


### Loss function ###

RMSE <- function(true_y, predicted_y){
  sqrt(mean((true_y - predicted_y)^2))
}
#1. global mean as sole predictor
global_mena_rmse <- RMSE(test$rating, global_mean)

#2. global_mean and genre effect

cat_values <- g$g
names(cat_values) <- g$genre

# Define a function to calculate the sum of values for each category in a string
cat_sum <- function(s) {
  # Split the string into individual categories
  categories <- strsplit(s, "\\|")[[1]]
  # Use the named numeric vector to lookup the values for each category
  values <- cat_values[categories]
  # Sum the values
  sum(values, na.rm = TRUE)
}

# Apply the function to each string in the character vector and store the results in a new vector
test_genre_sums <- sapply(test$genres, cat_sum)
test <- test %>% mutate(gs = unname(test_genre_sums))

genre_model_rmse <- RMSE(test$rating, (global_mean + test$gs))


### MOVIE EFFECTS ###

# Now we want to consider the primary cause and effect. How a movie, the subject of the rating, itself influences the rating. Are there movie effects? Let's see how the values are distributed.
train %>% group_by(movieId) %>% summarize(avg_rating = mean(rating), n = n()) %>%
  ggplot(aes(avg_rating))+
  geom_histogram()

train %>% group_by(movieId) %>% summarize(avg_rating = mean(rating), n = n()) %>% mutate(movieId = reorder(movieId, avg_rating)) %>%
  ggplot(aes(movieId, avg_rating))+
  geom_point(aes(size=n))
