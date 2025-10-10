#' A reparameterization in order to sample from the learned standard normal distribution of the VAE
#'
#' @param arg a layer of tensors representing the mean and variance
sampling_independent <- function(arg) {
  N_dims <- keras::k_int_shape(arg)[[2]] / 2
  z_mean <- arg[, 1:N_dims]
  z_log_var <- arg[, (N_dims + 1):(2 * N_dims)]
  b_size <- keras::k_int_shape(z_mean)[[1]]
  eps <- keras::k_random_normal(
    shape = c(b_size, op_cast(N_dims, dtype = 'int32')),
    mean = 0,
    stddev = 1
  )
  z_mean + tensorflow::tf$multiply(op_exp(z_log_var / 2), eps)
}

#' A custom kernel constraint function that restricts weights between the learned distribution and output. Nonzero weights are determined by the Q matrix.
#'
#' @param Q a binary matrix of size \code{N_dims} by \code{N_items}
#' @return returns a function whose parameters match keras kernel constraint format
q_constraint <- function(Q) {
  constraint <- function(w) {
    target <- w * Q
    diff = w - target
    w <- w * op_cast(k_equal(diff, 0), floatx())
    w * op_cast(k_greater_equal(w, 0), floatx())
  }
  constraint
}

#' A custom loss function for a VAE learning a standard normal distribution
#'
#' @param encoder the encoder model of the VAE, used to obtain z_mean and z_log_var from inputs
#' @param kl_weight weight for the KL divergence term
#' @param rec_dim the number of nodes in the input/output of the VAE
#' @return returns a function whose parameters match keras loss format
vae_loss_independent <- function(encoder, kl_weight, rec_dim) {
  loss <- function(input, output) {
    vals <- encoder(input)
    z_mean_val <- vals[[1]]
    z_log_var_val <- vals[[2]]
    kl_loss <- 0.5 *
      op_sum(
        op_square(z_mean_val) +
          op_exp(z_log_var_val) -
          1 -
          z_log_var_val,
        axis = -1L
      )
    rec_loss <- rec_dim * loss_categorical_crossentropy(input, output)
    op_mean(kl_weight * kl_loss + rec_loss)
  }
  loss
}