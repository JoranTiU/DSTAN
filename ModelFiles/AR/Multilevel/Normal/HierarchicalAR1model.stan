// The input data is a vector 'y' of length 'N', we have I individuals who we 
// measured T times for N (I*T) observations in total. We also have an 
// indicator variable that illustrates what individual (1,..., I) data belong 
// to, and what measurement occasion (1,..., T) the data was collected at. 
// Data is in long-format
data {
  int<lower=0> N; // number of observations
  int<lower=0> I; // Number of individuals
  int<lower=0> T; // Number of Timepoints
  int<lower = 1, upper = I> individual[N]; // integer vector the same length 
  // as the panel, indicating which individual each row corresponds to
  int<lower = 1, upper = T> time[N]; // integer vector with individual-time 
  //int<lower = 1, upper = N> IndT1[I];
  vector[N] y;
}

transformed data {

  // To make my priors universally applicable, I standardize the observed
  // data so I know the scale it's on.
  vector[N] y_std; // Vector to store the standardized scores
  real meanY; // The observed mean score
  real sdY; // the observed sd

  meanY = mean(y);
  sdY = sd(y);
  
  // standardizing based on observed mean and sd
  y_std = (y - meanY)/sdY;
}

// The parameters accepted by the model. 
parameters {
  // the hyper-parameters (mean and sd) of the means 
  real alpha_hat;
  real<lower = 0> alpha_scale;

  // Hyper-parameters for the overall AR parameter
  real<lower=-1, upper=1> beta_hat;
  real<lower = 0> beta_scale;
  
  // the individual means and AR-parameters
  vector[I] alpha;
  vector<lower=-1, upper=1>[I] beta;

  // the innovation-variance
  real<lower=0> sigma;
  
  
}

// The model to be estimated. We model the output 'y_std[n]' to be normally 
// distributed with mean 'alpha[n] + beta[n] * y_c[n-1]' and standard deviation 
// 'sigma'. We use the group-mean centered values of y as predictors so that 
// alpha gives us individual means instead of intercepts.
model {

  // vector to store latent-centered values
  vector[N] y_c;

  // priors for my parameters 
  alpha_hat ~ normal(0, 5);
  beta_hat ~ normal(0, .5);
  
  alpha_scale ~ normal(0, 1);
  beta_scale ~ normal(0, 1);
  sigma ~ normal(0, 2);
  
  
  for(i in 1:I) {
    alpha[i] ~ normal(alpha_hat, alpha_scale);
    
    beta[i] ~ normal(beta_hat, beta_scale);
    
    }
  
  
  // Create latent-centered values for T=1
  y_c[1] =  y_std[1] - alpha[individual[1]]; 
  

  // Analyse T > 1 based on the AR(1) model
  for (n in 2:N){
  
   // latent-mean centering scores at each occasion for T > 1  
   y_c[n] = y_std[n] - alpha[individual[n]];  

   // and the actual AR(1) model usiong the latent-centered predictors 
   if (time[n] > 1)
      y_std[n] ~ normal(
        alpha[individual[n]] + beta[individual[n]] * y_c[n-1], sigma);
  }
}

generated quantities {

  // Because I standardized, som parameters are in the "wrong" scale.
  // Here I transform back the individual means, the hyper-parameters for the
  // means and the innovation-variance to the original scale.

  vector[I] alphas_ind;
  real alpha_hat_raw;
  real<lower = 0> alpha_scale_raw; 
  real<lower = 0>  sigma_raw; 
  
  alphas_ind = (sdY * alpha) + meanY;
  alpha_hat_raw = (sdY * alpha_hat) + meanY;
  alpha_scale_raw = sdY*alpha_scale;
  
  sigma_raw = sigma*sdY;
  
}
