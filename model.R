#' Build the encoder for a VAE
#'
#' @param input_size an integer representing the number of items
#' @param layers a list of integers giving the size of each hidden layer
#' @param activations a list of strings, the same length as layers
#' @return two tensors: the input layer to the VAE and the last hidden layer of the encoder
build_hidden_encoder <- function(
  input_size,
  layers,
  activations = rep('softmax', length(layers)),
  batch_size
) {
  input <- layer_input(shape = c(input_size), name = 'input', batch_size = batch_size)
  h <- input
  if (length(layers) > 0) {
    for (layer in 1:length(layers)) {
      h <- layer_dense(
        h,
        units = layers[layer],
        activation = activations[layer],
        name = paste('hidden_', layer, sep = '')
      )
    }
  }
  list(input, h)
}

#' Build a VAE that fits to a standard N(0,I) latent distribution with independent latent traits
#'
#' @param N_items an integer giving the number of items on the assessment; also the number of nodes in the input/output layers of the VAE
#' @param N_dims an integer giving the number of skills being evaluated; also the dimensionality of the distribution learned by the VAE
#' @param Q a binary, \code{N_dims} by \code{N_items} matrix relating the assessment items with skills
#' @param enc_hid_arch a vector detailing the size of hidden layers in the encoder; the number of hidden layers is determined by the length of this vector
#' @param hid_enc_activations a vector specifying the activation function in each hidden layer in the encoder; must be the same length as \code{enc_hid_arch}
#' @param output_activation a string specifying the activation function in the output of the decoder; the ML2P model always uses 'sigmoid'
#' @param kl_weight an optional weight for the KL divergence term in the loss function
#' @param learning_rate an optional parameter for the adam optimizer
#' @return returns three keras models: the encoder, decoder, and vae.
build_vae_independent <- function(
  N_items,
  N_dims,
  Q,
  enc_hid_arch = c(ceiling((N_items + N_dims) / 2)),
  hid_enc_activations = rep('softmax', length(enc_hid_arch)),
  output_activation = 'softmax',
  kl_weight = 1,
  learning_rate = 0.001,
  batch_size
) {
  # weight_constraint <- q_constraint

  c(input, h) %<-% build_hidden_encoder(N_items, enc_hid_arch, hid_enc_activations, batch_size)

  h <- layer_embedding(h, input_dim  )
  z_mean <- layer_dense(h, units = N_dims, activation = 'linear', name = 'z_mean')
  z_log_var <- layer_dense(h, units = N_dims, activation = 'linear', name = 'z_log_var')
  z <- layer_lambda(layer_concatenate(list(z_mean, z_log_var)), sampling_independent)
  encoder <- keras_model(input, c(z_mean, z_log_var, z))

  latent_inputs <- layer_input(N_dims, name = 'latent_inputs', batch_size = batch_size)
  out <- layer_dense(
    latent_inputs,
    units = N_items,
    activation = output_activation,
    # kernel_constraint = weight_constraint(Q),
    name = 'vae_out'
  )
  decoder <- keras_model(latent_inputs, out)
  output <- decoder(encoder(input)[3]) ## get the z layer

  vae <- keras_model(input, output)
  vae_loss <- vae_loss_independent(encoder, kl_weight, N_items)

  compile(
    vae,
    optimizer = optimizer_adam(learning_rate = learning_rate),
    loss = vae_loss
  )

  list(encoder, decoder, vae)
}

#' Get trainable variables from the decoder, which serve as item parameter estimates.
#'
#' @param decoder a trained keras model; can either be the decoder or vae returned from \code{build_vae_independent()} or \code{build_vae_correlated}
#' @param model_type either 1 or 2, specifying a 1 parameter (1PL) or 2 parameter (2PL) model; if 1PL, then only the difficulty parameter estimates (output layer bias) will be returned; if 2PL, then the discrimination parameter estimates (output layer weights) will also be returned
#' @return a list which contains item parameter estimates; the length of this list is equal to model_type - the first entry in the list holds the difficulty parameter estimates, and the second entry (if 2PL) contains discrimination parameter estimates
get_item_parameter_estimates <- function(decoder) {
  all_decoder_weights <- get_weights(decoder)
  weights_length <- length(all_decoder_weights)
  
  list(
    all_decoder_weights[weights_length],
    all_decoder_weights[weights_length - 1]
  )
}

#' Feed forward response sets through the encoder, which outputs student ability estimates
#'
#' @param encoder a trained keras model; should be the encoder returned from either \code{build_vae_independent()} or \code{build_vae_correlated}
#' @param responses a \code{N_voters} by \code{N_items} matrix of binary responses, as used in training
#' @return a list where the first entry contains student ability estimates and the second entry holds the variance (or covariance matrix) of those estimates
get_ability_parameter_estimates <- function(encoder, responses) {
  encoded_responses <- encoder(responses)
  estimates_variances <- c()
  ability_parameter_estimates <- encoded_responses[[1]]
  ability_parameter_log_variance <- encoded_responses[[2]]
  if (ability_parameter_estimates$shape == ability_parameter_log_variance$shape) {
    estimates_variances[[1]] <- ability_parameter_estimates$numpy()
    estimates_variances[[2]] <- exp(ability_parameter_log_variance$numpy())
  } else {
    b <- tfprobability::tfb_fill_triangular(upper = FALSE)
    log_cholesky <- b$forward(ability_parameter_log_variance)
    cholesky <- tensorflow::tf$linalg$expm(log_cholesky)
    cov_matrices <- tensorflow::tf$matmul(
      cholesky,
      tensorflow::tf$transpose(cholesky, c(0L, 2L, 1L))
    )
    estimates_variances[[1]] <- ability_parameter_estimates$numpy()
    estimates_variances[[2]] <- cov_matrices
  }
  estimates_variances
}
