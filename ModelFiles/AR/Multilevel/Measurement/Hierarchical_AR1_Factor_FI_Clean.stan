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
  vector[N] rawLatent; // if AR the latent-score differs across time so N 
  // instead of I
  vector[P-1] means; // intercepts
  vector[P-1] rawLoadings; // P-1 loadings
  vector<lower=0>[P] rload;// residuals
  
  // AR parameters
  real alpha_hat;
  real<lower = 0> alpha_scale;
  real<lower=-1, upper=1> beta_hat;
  real<lower = 0> beta_scale;
  real<lower=0> sigma;
  vector[I] alpha;
  vector<lower=-1, upper=1>[I] beta;
}

transformed parameters {
  vector[P] perRowMean[N];
  cov_matrix[P] cov1;
  cholesky_factor_cov[P] cf;
  
  for(j in 1:N){

    perRowMean[j,1] = rawLatent[j] * 1; # Fix first loading to 1 and intercept 
    //to 0
    perRowMean[j,2] = means[1] + rawLatent[j] * rawLoadings[1];
    perRowMean[j,3] = means[2] + rawLatent[j] * rawLoadings[2];
    perRowMean[j,4] = means[3] + rawLatent[j] * rawLoadings[3];
    perRowMean[j,5] = means[4] + rawLatent[j] * rawLoadings[4];
  }
  

  // only need to add rawLoadings * rawLoadings' to cov1 for a marginal likelihood
  cov1 = diag_matrix(rload .* rload);
  cf = cholesky_decompose(cov1);
}

model {
  vector[N] lat_c;

  rawLoadings ~ normal(0,2);
  rload ~ normal(0,1);
  means ~ normal(0,2);

  alpha_hat ~ normal(0, 5);
  beta_hat ~ normal(0, .5);
  alpha_scale ~ normal(0, 1);
  beta_scale ~ normal(0, 1);
  sigma ~ normal(0, 2);
  
  for(i in 1:I) {
    
    alpha[i] ~ normal(alpha_hat, alpha_scale);
    beta[i] ~ normal(beta_hat, beta_scale);
    
  }
  
  
  //if (loadingsAR[1] < 0){ 
  //  loadingsAR = -loadingsAR;
  //  latentsAR = -latentsAR;
  //}
  
  // person-mean center the latent-score to get means instead of intercept
  lat_c[1] =  rawLatent[1] - alpha[individual[1]];
  
  
  for (n in 2:N){
  
   lat_c[n] = rawLatent[n] - alpha[individual[n]];  
    
   if (time[n] > 1)
      //sigma = 1 - pow(beta[individual[n]],2);
      //latentsAR[n] ~ normal(alpha[individual[n]] + beta[individual[n]] * lat_c[n-1], sigma);
      rawLatent[n] ~ normal(alpha[individual[n]] + beta[individual[n]] * lat_c[n-1], sigma);
  }
  
  Y ~ multi_normal_cholesky(perRowMean, cf);
  
}


generated quantities {
  vector[P-1] loadings = rawLoadings;
  //vector[N] latents = to_vector(rawLatent);
  vector[I] alpha_t = to_vector(alpha);
  //vector[I] beta_t = to_vector(beta);
  real alpha_hat_t = alpha_hat;
  //real beta_hat_t = beta_hat;
  vector[P] resid;
  resid = rload .* rload;

  if (loadings[1] < 0){ 
    loadings = -loadings;
    //latents = -latents;
    alpha_t = -alpha;
    //beta_t = -beta;
    alpha_hat_t = -alpha_hat;
    //beta_hat_t = -beta_hat;
    
  }
}
