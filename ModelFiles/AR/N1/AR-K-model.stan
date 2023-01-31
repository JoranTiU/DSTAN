// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> K;
  int<lower=0> N;
  real y[N];
}

// The parameters accepted by the model. Our model accepts two parameters 'mu' and 'sigma'.
parameters {
  real alpha;
  real<lower=-1, upper=1> beta[K];
  real<lower=0> sigma;
}

// The model to be estimated. We model the output 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, .7);
  sigma ~ normal(0, 2);
  
  for (n in (K+1):N) {
    real mu = alpha;
    for (k in 1:K)
      mu += beta[k] * y[n-k];
      y[n] ~ normal(mu, sigma);
  }
}
