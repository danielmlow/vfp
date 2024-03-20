library(ggplot2)  # For plotting
library(dplyr)    # For data manipulation

df_speech <- read.csv('/Users/danielmlow/Dropbox (MIT)/datum/vfp/data/input/features/egemaps_vector_speech.csv')

# Assuming df_speech is your dataframe
variables <- colnames(df_speech)[2:(ncol(df_speech) - 4)]
length(variables)

df_speech <- df_speech[variables]

unique_pairs <- combn(variables, 2, simplify = FALSE)
length(unique_pairs)


# Remove outliers
outlier_sd <- 3

df_speech <- df_speech %>%
  mutate(across(everything(), ~ifelse(. > mean(., na.rm = TRUE) + outlier_sd * sd(., na.rm = TRUE) | 
                                        . < mean(., na.rm = TRUE) - outlier_sd * sd(., na.rm = TRUE), NA, .)))


# Loop through each pair (subset to every 100th pair if desired)
for(i in seq(3, length(unique_pairs), by = 99)) {
  pair <- unique_pairs[[i]]
  ggplot(df_speech, aes_string(x=pair[1], y=pair[2]))+
    geom_point(alpha = 0.3)+
    geom_jitter()+
    geom_smooth()

  
  # Print the plot
  print(last_plot())
}





