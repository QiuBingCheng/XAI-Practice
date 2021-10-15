if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()
library(tensorflow)
library(keras)
K <- keras::backend()

# Parameters --------------------------------------------------------------

batch_size <- 100
original_dim <- 784
latent_dim <- 2
intermediate_dim_1 <- 392
intermediate_dim_2 <-178
intermediate_dim_3 <- 85
intermediate_dim_4 <- 30
epochs <- 30
epsilon_std <- 1.0
# Model definition --------------------------------------------------------

encoder_input <- layer_input(shape = c(original_dim))
h1 <- layer_dense(encoder_input, intermediate_dim_1, activation = "relu",name="encoder_h1")
h2 <- layer_dense(h1, intermediate_dim_2, activation = "relu",name="encoder_h2")
h3 <- layer_dense(h2, intermediate_dim_3, activation = "relu",name="encoder_h3")
h4 <- layer_dense(h3, intermediate_dim_4, activation = "relu",name="encoder_h4")
z_mean <- layer_dense(h4, latent_dim,name="z_mean")
z_log_var <- layer_dense(h4, latent_dim,name="z_log_var")

sampling <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]), 
    mean=0.,
    stddev=epsilon_std
  )
  
  z_mean + k_exp(z_log_var/2)*epsilon
}

# note that "output_shape" isn't necessary with the TensorFlow backend
z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling)
# we instantiate these layers separately so as to reuse them later
decoder_h1 <- layer_dense(units = intermediate_dim_4, activation = "relu",name="decoder_h1")
decoder_h2 <- layer_dense(units = intermediate_dim_3, activation = "relu",name="decoder_h2")
decoder_h3 <- layer_dense(units = intermediate_dim_2, activation = "relu",name="decoder_h3")
decoder_h4 <- layer_dense(units = intermediate_dim_1, activation = "relu",name="decoder_h4")
decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid",name="decoder_output")
h1_decoded <- decoder_h1(z)
h2_decoded <-decoder_h2((h1_decoded))
h3_decoded <-decoder_h3((h2_decoded))
h4_decoded <-decoder_h4((h3_decoded))
x_decoded_mean_4 <- decoder_mean(h4_decoded)

# end-to-end autoencoder
vae <- keras_model(encoder_input, x_decoded_mean_4)
vae
# encoder, from inputs to latent space
encoder <- keras_model(encoder_input, z_mean)
encoder
# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = latent_dim)
h_decoded_1 <- decoder_h1(decoder_input)
h_decoded_2 <- decoder_h2(h_decoded_1)
h_decoded_3 <- decoder_h3(h_decoded_2)
h_decoded_4 <- decoder_h4(h_decoded_3)
x_decoded_mean_4 <- decoder_mean(h_decoded_4)
generator <- keras_model(decoder_input, x_decoded_mean_4)
generator

vae_loss <- function(encoder_input, x_decoded_mean_4){
  xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(encoder_input, x_decoded_mean_4)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  xent_loss + kl_loss
}

vae %>% compile(optimizer = "rmsprop", loss = vae_loss)



# Data preparation --------------------------------------------------------

mnist <- dataset_mnist()
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255
x_train <- array_reshape(x_train, c(nrow(x_train), 784), order = "F")
x_test <- array_reshape(x_test, c(nrow(x_test), 784), order = "F")
table(x_train)
x_train[1,]
# Model training ----------------------------------------------------------

vae %>% fit(
  x_train, x_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
  validation_data = list(x_test, x_test)
)


# Visualizations ----------------------------------------------------------

library(ggplot2)
library(dplyr)
x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)

x_test_encoded %>%
  as_data_frame() %>% 
  mutate(class = as.factor(mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()

# display a 2D manifold of the digits
n <- 20  # figure with 15x15 digits
digit_size <- 28

# we will sample n points within [-4, 4] standard deviations
grid_x <- seq(-3, 3, length.out = n)
grid_y <- seq(-3, 3, length.out = n)

rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = 28) )
  }
  rows <- cbind(rows, column)
}
rows %>% as.raster() %>% plot()



