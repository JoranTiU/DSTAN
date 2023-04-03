// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//
// The input data is a vector 'y' of length 'N'.
functions {
  vector lp_reduce( vector beta , vector theta , real[] xr , int[] xi ) {
    int n = 70; // size(xr); // Number of cases per shard
    int K = 2;
    vector[K] Y_std[n];
    vector[K] Xc[n];
    vector[K] Mean[n];
    real lp;
    
    real y[n] = xr[1:n]; // Scores on DV 1
    real y2[n] = xr[(n+1):(2*n)]; // Scores on DV 2
    real X[n] = xr[((2*n)+1):(3*n)]; // Centered Scores on DV 1, used as predictor
    real X2[n] = xr[((3*n)+1):(4*n)];// Centered Scores on DV 2, used as predictor
    
    vector[n] alpha = theta[1:n]; // individual means on DV 1
    vector[n] alpha2 = theta[(n+1):(2*n)]; // individual means on DV 2
    vector[n] beta1 = theta[((2*n)+1):(3*n)]; // AR  parameter for DV 1
    vector[n] beta2 = theta[((3*n)+1):(4*n)]; // AR parameter for DV 2
    vector[n] beta3 = theta[((4*n)+1):(5*n)]; // Cross-lagged effects from DV 2 to Dv 1
    vector[n] beta4 = theta[((5*n)+1):(6*n)]; // Cross-lagged effects from DV 1 to Dv 2
    
    matrix[2,2] L_Omega = to_matrix(beta[1:4],2,2); 
    
    
    Y_std[1:n,1] = y[1:n];
    Y_std[1:n,2] = y2[1:n];
    
    Xc[1:n,1] = to_array_1d(to_vector(X[1:n]) - alpha[1:n]);
    Xc[1:n,2] = to_array_1d(to_vector(X2[1:n]) - alpha2[1:n]);
    
    Mean[,1] = to_array_1d(alpha[1:n] + to_vector(beta1 .* to_vector(Xc[1:n,1])) + to_vector(beta3 .* to_vector(Xc[1:n,2])));
    Mean[,2] = to_array_1d(alpha2[1:n] + to_vector(beta2 .* to_vector(Xc[1:n,2])) + to_vector(beta4 .* to_vector(Xc[1:n,1])));
    
    lp = multi_normal_cholesky_lpdf(Y_std[2:n]|Mean[2:n], L_Omega); 
    return [lp]';
  }
} 
data {
  int N; // number of observations (number of rows in the panel) this is equal to N*T due to long format
  int K; // number of dimensions of the outcome variable
  int I; // number of individuals
  int T; // The greatest number of time periods for any individual
  int<lower = 1, upper = I> individual[N]; // integer vector the same length 
  // as the panel, indicating which individual each row corresponds to
  int<lower = 1, upper = T> time[N]; // integer vector with individual-time 
  // (not absolute time). Each individual starts at 1
  vector[K] Y[N]; // Dependent variables
  vector[K] X[N]; // lagged predictor
}
transformed data {
  // Standardize data to speed up computation
  vector[K] Y_std[N]; 
  vector[K] X_std[N];
  
  real meanY[K];
  real sdY[K];
  
  real meanX[K];
  real sdX[K];
  
  // Transformations to get data into shards
  // 200 shards. Run per individual and combine
  // M = N/200 = 14000/200 = 70
  int n_shards = I;
  int M = T;// N/n_shards;
  int xi[n_shards,M];  
  real xr[n_shards, 4*M];// 4M because two variables, and 2 lagged predicted means get stacked in array
  // split into shards
  
  meanY[1] = mean(Y[,1]);
  meanY[2] = mean(Y[,2]);
  
  sdY[1] = sd(Y[,1]);
  sdY[2] = sd(Y[,2]);
  
  meanX[1] = mean(X[,1]);
  meanX[2] = mean(X[,2]);
  
  sdX[1] = sd(X[,1]);
  sdX[2] = sd(X[,2]);
  
  Y_std[,1] = to_array_1d((to_vector(Y[,1]) - mean(Y[,1]))/sd(Y[,1]));
  Y_std[,2] = to_array_1d((to_vector(Y[,2]) - mean(Y[,2]))/sd(Y[,2]));
  
  X_std[,1] = to_array_1d((to_vector(X[,1]) - mean(X[,1]))/sd(X[,1]));
  X_std[,2] = to_array_1d((to_vector(X[,2]) - mean(X[,2]))/sd(X[,2]));
  
  
  for ( i in 1:n_shards ) {
    int j = 1 + (i-1)*M;
    int k = i*M;
    xi[i,1:M] = individual[ j:k ];
    
    xr[i,1:M] = Y_std[j:k, 1 ];
    xr[i,(M+1):(2*M)] = Y_std[j:k, 2 ];
    xr[i,((2*M)+1):(3*M)] = X_std[j:k, 1 ];
    xr[i,((3*M)+1):(4*M)] = X_std[j:k, 2 ];
    
  }
  
  
}

// The parameters accepted by the model. 
parameters {
  vector[K] tau_int;
  //vector[K] tau;
  real<lower = -1, upper = 1> corr;
  vector[K*K] beta_hat_location; // <lower = -1, upper = 1> means of the VAR(1) parameters
  vector<lower = 0>[K*K] beta_hat_scale; // Sd's of the  AR and Cross-lagged products
  vector[K] alpha_hat_location; // means for the K DV's
  vector<lower = 0>[K] alpha_hat_scale;// sd's of the means of predictors
  real<lower = -1, upper = 1> corr_alpha;////vector<lower = -1, upper = 1>[[(K*(K-1))/2]] corr_alpha;
  
  
  // individual-level parameters
  matrix[I,K*K] z_beta;
  matrix[I,K] z_alpha;
  
  corr_matrix[K*K] Omega_beta;
}

transformed parameters {
  // want to add exponent of tau and draw tau's in same way as alpha's and beta's, from multivariate distribution
  matrix[K, K] betas[I]; // individual VAR(1) coefficient matrix
  vector[K] alpha[I]; // individual means matrix
  vector[K] tau; 
  matrix[K,K] Omega;
  matrix[K,K] Omega_alpha;
  
  matrix[K, K] L_alpha;
  matrix[K, K] L_Omega;
  matrix[(K*K), (K*K)] L_beta;
  
  Omega_alpha = diag_matrix(rep_vector(1,K));
  Omega_alpha[1,2] = corr_alpha;
  Omega_alpha[2,1] = corr_alpha;
  
  Omega = diag_matrix(rep_vector(1,K));
  Omega[1,2] = corr;
  Omega[2,1] = corr;
  
  
  // Transform tau to make lognormal
  tau = exp(tau_int);
   
  L_Omega = cholesky_decompose(quad_form_diag(Omega, tau));  
  L_alpha = cholesky_decompose(quad_form_diag(Omega_alpha, alpha_hat_scale));
  L_beta = cholesky_decompose(quad_form_diag(Omega_beta, beta_hat_scale)); // some rejections occurs here
 
  for(i in 1:I) { // Can't remove this loop, because I need I alpha and beta's in output
    alpha[i] = alpha_hat_location + L_alpha * z_alpha[i]';
    // implies: alpha[i] ~ multi_normal(alpha_hat_location, alpha_hat_scale)
    betas[i] = to_matrix((beta_hat_location + L_beta * z_beta[i]'), K, K);
    
    }
}

// The model to be estimated.
model {
  // hyperpriors
  // To get mean of 3 and var of 1 with lognormal set normal mean to -0.111572 and sd to 0.4723812
  // for mean(var) inno's of 3(1), 5(1), 5(2), and 7(3), normal mean and sd vary between 1 and 2, and .04 and .10
  // respectively
  alpha_hat_location ~ normal(0, 5); // average mean between 0 and 6, seems reasonable given most likert-scales used 
  // in social science
  alpha_hat_scale ~ normal(0, 1); // SD of means (SE's) between 0 and 4 (half-normal)
  beta_hat_location ~ normal(0, 0.5);// AR and cross-lagged parameters between -.5 and .5
  beta_hat_scale ~ normal(0.5, 0.5);// SE's of AR- and cross-lagged between 0 and .6 (half-normal)
  tau_int ~ normal(0, 5);
  //tau ~ normal(0, 5);
  corr ~ normal(0, 0.7);// Correlation between residuals between -7 and .7
  corr_alpha ~ normal(0, 0.7); // correlation between means between -.7 and .7
  
  Omega_beta ~ lkj_corr(2);
  
  // hierarchical priors
  // non-centered parameterization
  to_vector(z_alpha) ~ std_normal();
  to_vector(z_beta) ~ std_normal();
 
  
  
  {
  
  vector[K*K] beta;
  vector[6*M] theta[n_shards]; // 8M because two means, two AR-prs, 2 cl-pars, Omega and tau, 6 if Omega and tau fixed
 
  
  for ( i in 1:n_shards ) {
    int j = 1 + (i-1)*M;
    int k = i*M;
    
    theta[i,1:M] = to_vector(alpha[individual[j:k],1]);
    theta[i,(M+1):(2*M)] = to_vector(alpha[individual[j:k],2]);
    theta[i,((2*M)+1):(3*M)] = to_vector(betas[individual[j:k],1,1]);
    theta[i,((3*M)+1):(4*M)] = to_vector(betas[individual[j:k],K,K]);
    theta[i,((4*M)+1):(5*M)] = to_vector(betas[individual[j:k],1,K]);
    theta[i,((5*M)+1):(6*M)] = to_vector(betas[individual[j:k],K,1]);
  }
  
  
  beta = to_vector(L_Omega);
  
   target += sum(map_rect(lp_reduce, beta, theta, xr, xi));
  
  }

}

generated quantities {
  matrix[K, K] betas_raw[I]; // individual VAR(1) coefficient matrix
  vector[K] alphas_raw[I]; // individual means matrix
  
  vector[K] alpha_hat_location_raw;
  vector[K] alpha_hat_scale_raw; 
  matrix[K, K] beta_hat_location_raw;
  matrix[K, K] beta_hat_scale_raw;
  
  vector[K] tau_raw; 
  
  alphas_raw[,1] = to_array_1d((sdY[1] * (to_vector(alpha[,1]))) + meanY[1]);
  alphas_raw[,2] = to_array_1d((sdY[2] * (to_vector(alpha[,2]))) + meanY[2]);
  
  betas_raw[,1,1] = to_array_1d(to_vector(betas[,1,1]) * sdY[1]/sdX[1]);
  betas_raw[,1,2] = to_array_1d(to_vector(betas[,1,2]) * sdY[1]/sdX[2]);
  betas_raw[,2,1] = to_array_1d(to_vector(betas[,2,1]) * sdY[2]/sdX[1]);
  betas_raw[,2,2] = to_array_1d(to_vector(betas[,2,2]) * sdY[2]/sdX[2]);
  
  tau_raw[1] = exp(tau_int[1])*sdY[1];
  tau_raw[2] = exp(tau_int[2])*sdY[2];
  
  //tau_raw[1] = tau[1]*sdY[1];
  //tau_raw[2] = tau[2]*sdY[2];
  
  alpha_hat_location_raw[1] = (sdY[1] *  alpha_hat_location[1]) + meanY[1];
  alpha_hat_location_raw[2] = (sdY[2] *  alpha_hat_location[2]) + meanY[2];
  
  alpha_hat_scale_raw[1] = sdY[1]*alpha_hat_scale[1];//sdY[1]^2*alpha_hat_scale[1]
  alpha_hat_scale_raw[2] = sdY[2]*alpha_hat_scale[2];
  
  beta_hat_location_raw[1,1] = beta_hat_location[1]*sdY[1]/sdX[1];
  beta_hat_location_raw[2,1] = beta_hat_location[2]*sdY[2]/sdX[1];
  beta_hat_location_raw[1,2] = beta_hat_location[3]*sdY[1]/sdX[2];
  beta_hat_location_raw[2,2] = beta_hat_location[4]*sdY[2]/sdX[2];
  
  beta_hat_scale_raw[1,1] = (sdY[1]/sdX[1])*beta_hat_scale[1];
  beta_hat_scale_raw[2,1] = (sdY[2]/sdX[1])*beta_hat_scale[2];
  beta_hat_scale_raw[1,2] = (sdY[1]/sdX[2])*beta_hat_scale[3];
  beta_hat_scale_raw[2,2] = (sdY[2]/sdX[2])*beta_hat_scale[4];
  
  
 
}

