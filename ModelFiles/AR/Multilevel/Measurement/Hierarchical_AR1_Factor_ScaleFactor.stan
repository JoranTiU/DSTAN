data {
  int<lower = 0> N; // number of observations
  int<lower=1> P; // Number of indicators
  int<lower=0> I; // Number of individuals
  int<lower=0> T; // Number of Timepoints
  
  // Indicator Functions
  int<lower = 1, upper = I> individual[N]; // integer vector the same length 
  // as the panel, indicating which individual each row corresponds to
  int<lower = 1, upper = T> time[N]; // integer vector with individual-time 
  //int<lower = 1, upper = N> IndT1[I];
  vector[P] Y[N];
}


parameters {
  vector[N] rawLatent; // The latent-score at each measurement occasion and 
  // for every person (so I*T = N different scores)
  vector[P] means; // the intercepts
  vector[P] rawLoadings; // the factor-loadings
  vector<lower=0>[P] rload;// the residuals (I use a Cholesky decomposition
  // so will turn into residual in a separate step)
  
  // Hyper-parameters for the intercepts (mean and sd)
  vector[P] alpha_means;
  vector<lower = 0>[P] sd_means;
  
  // Hyper-parameters for the overall AR parameter
  real<lower=-1, upper=1> beta_hat;
  real<lower = 0> beta_scale;

  // the individual AR-parameters
  vector<lower=-1, upper=1>[I] beta;
}

transformed parameters {

  // parameters for the observed scores based on the measurement model, and
  // for the Cholseky-decomposed residuals  
  vector[P] perRowMean[N];
  cov_matrix[P] cov1;
  cholesky_factor_cov[P] cf;

  // Here we create the (predicted) observed scores based on the Measurement
  // Model  
  for(j in 1:N){
    perRowMean[j,1] = means[1] + rawLatent[j] * rawLoadings[1] ;
    perRowMean[j,2] = means[2] + rawLatent[j] * rawLoadings[2] ;
    perRowMean[j,3] = means[3] + rawLatent[j] * rawLoadings[3] ;
    perRowMean[j,4] = means[4] + rawLatent[j] * rawLoadings[4] ;
    perRowMean[j,5] = means[5] + rawLatent[j] * rawLoadings[5] ;
    
  }
  

  // The things needed to estimate the residuals using Cholesky decomposition 
  // only need to add rawLoadings * rawLoadings' to cov1 for a marginal 
  // likelihood
  cov1 = diag_matrix(rload .* rload);
  cf = cholesky_decompose(cov1);

}

model {
  
  // Priors for my parameters
  rawLoadings ~ normal(0,2);
  rload ~ normal(0,1);
  alpha_means ~ normal(0,2);
  sd_means ~ normal(0,2);
  means ~ normal(alpha_means,sd_means);
  beta_hat ~ normal(0, .5);
  beta_scale ~ normal(0, 1);

  for(i in 1:I) {
    
    beta[i] ~ normal(beta_hat, beta_scale);
    
  }
  

  // The model: 
  // First the structural model with the AR-process on the latent-scores
  for (n in 2:N){


  // The variance at T1 in an AR(1) model has an expression which is use
  // to sample proper T1 estimates  
  if (time[n] == 1){
   rawLatent[n] ~ normal(0, sqrt(
      (1/(1 - pow(beta[individual[n]],2)))
     )
    ); 
   } 
  
  // Then I sample the rest based on an AR(1) Model
   if (time[n] > 1){
      // fixing inno var to 1
      rawLatent[n] ~ normal(beta[individual[n]] * rawLatent[n-1], 1);
   }
   
  }
  
  // Second the Measurement Model that related the actual observed scores
  // to the factor-model
  Y ~ multi_normal_cholesky(perRowMean, cf);
  
   
}


generated quantities {
  //vector[P] loadings = rawLoadings;
  //vector[I] beta_t = to_vector(beta);
  //real beta_hat_t = beta_hat;

  // Turn Cholesky back into actual residuals
  vector[P] resid;
  resid = rload .* rload;
 
  //if (loadings[1] < 0){ 
  //  loadings = -loadings;
  //  //beta_t = -beta;
  //  //beta_hat_t = -beta_hat;
  //}
}
