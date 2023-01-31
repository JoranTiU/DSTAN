// The input data is a vector 'y' of length 'N', we I individuals who we measured T times for 
// N (I*T) observations in total. We also have an indicator variable that illustrates what 
// individual (1,..., I) data belong to, and what measurement occasion (1,..., T) the data was 
// collected at. In addition, we provide the number of missing and observed datapoints, an 
// indicator for which rows in the input data are missing, an indicator for which rows
// are observed and a vector with all observed scores on the DV
data {
  int<lower=0> N;
  int<lower=0> I;
  int<lower=0> T;
  int<lower = 1, upper = I> individual[N]; 
  int<lower = 1, upper = T> time[N]; 
  int<lower = 1, upper = N> IndLike[N-I]; 
  int<lower = 1, upper = N> IndLikeLag[N-I]; 
  //int<lower = 1, upper = N> IndT1[I]; 
 
  
  vector[N] y_comp;
}

//transformed data {
//  vector[N] y_std;
//  real meanY;
//  real sdY;

//  meanY = mean(y_comp);
//  sdY = sd(y_comp);

//  y_std = (y_comp - meanY)/sdY;
//}

// The parameters accepted by the model. 
parameters {
  real alpha_hat;
  real<lower = 0> alpha_scale;
  real<lower=-1, upper=1> beta_hat;
  real<lower = 0> beta_scale;
  
  vector[I] alpha;
  vector<lower=-1, upper=1>[I] beta;
  real<lower=0> sigma;
  real<lower=0> sd_meas;
  
  //vector[N_miss] y_miss;
  
  // The below parameters are for the measurement-error free true-scores
  vector[N] y_lat;
  
  
  
  
}

// The model to be estimated. We model the output 'y_std[n]' to be normally distributed with mean 
//'alpha[n] + beta[n] * y_c[n-1]' and standard deviation 'sigma'. We use the group-mean centered
// values of y as predictors so that alpha gives us individual means instead of intercepts.
model {
  vector[N] y_c;
   
  alpha_hat ~ normal(0, 5);
  beta_hat ~ normal(0, .5);
  
  alpha_scale ~ normal(0, 1);
  beta_scale ~ normal(0, 1);
  sigma ~ normal(0, .5);
  //sigma ~ normal(0, 2); //sigma ~ normal(0, 2); sigma ~ normal(0, 1); sigma ~ normal(0, .5);
  //sigma ~ normal(0, .4);
  
  // Determine true-scores based on prior and observed scores
  sd_meas ~ normal(0, .3);
  //sd_meas ~ normal(0.5, .1);
  //sd_meas ~ normal(0, .3); //sd_meas ~ normal(0, 2); //sd_meas ~ normal(0, 1); sd_meas ~ normal(0, .5) 
  
  
  for(i in 1:I) {
    alpha[i] ~ normal(alpha_hat, alpha_scale);
    
    beta[i] ~ normal(beta_hat, beta_scale);
    
    }
  
  
  y_c =  y_lat - alpha[individual]; 
  
  y_lat[IndLike] ~ normal(alpha[individual[IndLike]] + beta[individual[IndLike]] .* y_c[IndLikeLag], sigma);

  // measurement model
  y_comp ~ normal(y_lat, sd_meas);
  
 
  
}

//generated quantities {
  //vector[I] alphas_ind;
  //vector[N] y_lat_raw;
  
  //real alpha_hat_raw;
  //real<lower = 0> alpha_scale_raw; 
  //real<lower = 0>  sigma_raw; 
  //real<lower = 0>  sd_meas_raw; 
  
  //alphas_ind = (sdY * alpha) + meanY;
  //alpha_hat_raw = (sdY * alpha_hat) + meanY;
  //alpha_scale_raw = sdY*alpha_scale;
  
  //y_lat_raw = (sdY * y_lat) + meanY;
  
  
  //sigma_raw = sigma*sdY;
  //sd_meas_raw = sd_meas*sdY;
  
//}
