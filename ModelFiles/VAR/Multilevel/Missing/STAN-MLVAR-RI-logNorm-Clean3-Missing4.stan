// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//
// The input data is a vector 'y' of length 'N'.
functions {
  vector lp_reduce( vector beta , vector theta , real[] xr , int[] xi ) {
    int n = 100; // size(xr); // Number of cases per shard
    int K = 2;
    vector[K] Y_std[n];
    vector[K] Xc[n];
    vector[K] Mean[n];
    matrix[K,K] L_Omega;
    real lp;
    
    real y[n] = to_array_1d(theta[1:n]); // Scores on DV 1
    real y2[n] = to_array_1d(theta[(n+1):(2*n)]); // Scores on DV 2
    real X[n] = to_array_1d(theta[((2*n)+1):(3*n)]); // Centered Scores on DV 1, used as predictor
    real X2[n] = to_array_1d(theta[((3*n)+1):(4*n)]);// Centered Scores on DV 2, used as predictor
    
    vector[n] alpha = theta[((4*n)+1):(5*n)]; // individual means on DV 1
    vector[n] alpha2 = theta[((5*n)+1):(6*n)]; // individual means on DV 2
    vector[n] beta1 = theta[((6*n)+1):(7*n)]; // AR  parameter for DV 1
    vector[n] beta2 = theta[((7*n)+1):(8*n)]; // AR parameter for DV 2
    vector[n] beta3 = theta[((8*n)+1):(9*n)]; // Cross-lagged effects from DV 2 to Dv 1
    vector[n] beta4 = theta[((9*n)+1):(10*n)]; // Cross-lagged effects from DV 1 to Dv 2
    matrix[K,K] Omega = to_matrix(theta[((10*n)+1):(((10*n)+1)+3)],K,K); // The correlation matrix for the person in this shard.
    vector[K] tau = theta[((10*n)+5):(((10*n)+5)+1)]; // The sd's for the Y's for the person in this shard.
    
    Y_std[1:n,1] = y[1:n];
    Y_std[1:n,2] = y2[1:n];
    
    Xc[1:n,1] = to_array_1d(to_vector(X[1:n]) - alpha[1:n]);
    Xc[1:n,2] = to_array_1d(to_vector(X2[1:n]) - alpha2[1:n]);
    
    Mean[,1] = to_array_1d(alpha[1:n] + to_vector(beta1 .* to_vector(Xc[1:n,1])) + to_vector(beta3 .* to_vector(Xc[1:n,2])));
    Mean[,2] = to_array_1d(alpha2[1:n] + to_vector(beta2 .* to_vector(Xc[1:n,2])) + to_vector(beta4 .* to_vector(Xc[1:n,1])));
    
    L_Omega = cholesky_decompose(quad_form_diag(Omega, tau));
 
    lp = multi_normal_cholesky_lpdf(Y_std[2:n]|Mean[2:n], L_Omega); 
    return [lp]';
  }
} 
data {
  int N; // number of observations (number of rows in the panel) this is equal to N*T due to long format
  //int Ncomp; // 
  int K; // number of dimensions of the outcome variable
  int I; // number of individuals
  int T; // The greatest number of time periods for any individual
  int<lower = 1, upper = I> individual[N]; // integer vector the same length 
  // as the panel, indicating which individual each row corresponds to
  int<lower = 1, upper = T> time[N]; // integer vector with individual-time 
  // (not absolute time). Each individual starts at 1
  //vector[K] Yobs[N]; // Dependent variables
  //vector[K] Xobs[N]; // lagged predictor
  vector[K] Yfill[N]; // Dependent variables
  vector[K] Xfill[N]; // lagged predictor
  //int<lower = 0> N_obs;
  int<lower = 0> N_mis[K];
  int<lower = 0> N_mis_X[K];
  int<lower = 0> N_comp[K];
  int<lower = 0> N_comp_X[K];
  vector[N_comp[1]] Ycomp1; 
  vector[N_comp[2]] Ycomp2;
  vector[N_comp_X[1]] Xcomp1;
  vector[N_comp_X[2]] Xcomp2;
  //vector[N_comp[1]] Xcomp1;
  //vector[N_comp[2]] Xcomp2;
  int<lower = 0> miss_ind1[N_mis[1]];
  int<lower = 0> miss_ind2[N_mis[2]];
  int<lower = 0> miss_ind1_x[N_mis_X[1]];
  int<lower = 0> miss_ind2_x[N_mis_X[2]];
  int<lower = 0> comp_ind1[N_comp[1]];
  int<lower = 0> comp_ind2[N_comp[2]];
  int<lower = 0> comp_ind1_x[N_comp_X[1]];
  int<lower = 0> comp_ind2_x[N_comp_X[2]];
  int<lower = 0> miss_ind1_xm1[N_mis_X[1]];
  int<lower = 0> miss_ind2_xm1[N_mis_X[2]];
  //int<lower = 1, upper = N_obs + N_mi
  //int<lower = 1, upper = N_obs + N_mis> ii_obs[N_obs];
  //int<lower = 1, upper = N_obs + N_mis> ii_mis[N_mis];
  //real y_obs[N_obs];
  
  //matrix[N, K] Y_obs; // Matrix with 1 for not missing and 0 for missing
  //int<lower = 0, upper = N> Num_Missing[N,K];
}

transformed data {
  // Standardize data to speed up computation
  real meanY[K];
  real sdY[K];
  
  real meanX[K];
  real sdX[K];
  
  int n_shards = I;
  int M = T;// N/n_shards;
  int xi[n_shards,M];
  real xr[n_shards,2];// 4M because two variables, and 2 lagged predicted means get stacked in array
  // split into shards
  
  for ( i in 1:I ) {
    int j = 1 + (i-1)*T;
    int k = i*T;
    xi[i,1:T] = individual[ j:k ];
  }
  
    
  // Mae equal to 100
  xr[,1] = to_array_1d(Yfill[1:I,1]);
  xr[,2] = to_array_1d(Yfill[1:I,2]);
  //xr[3,3] = 1;
  //xr[4,4] = 1;
  
  
  meanY[1] = mean(Ycomp1);
  meanY[2] = mean(Ycomp2);
  
  sdY[1] = sd(Ycomp1);
  sdY[2] = sd(Ycomp2);
  
  meanX[1] = mean(Xcomp1);
  meanX[2] = mean(Xcomp2);
  
  sdX[1] = sd(Xcomp1);
  sdX[2] = sd(Xcomp2);
  

  
}

// The parameters accepted by the model. 
parameters {
  vector[K] tau_hat_location;
  vector<lower =0>[K] tau_hat_scale;
  real<lower = -1, upper = 1> corr_tau;
  real<lower = -1, upper = 1> c_location;
  real<lower =0> c_scale;
  vector[K*K] beta_hat_location; // <lower = -1, upper = 1> means of the VAR(1) parameters
  vector<lower = 0>[K*K] beta_hat_scale; // Sd's of the  AR and Cross-lagged products
  vector[K] alpha_hat_location; // means for the K DV's
  vector<lower = 0>[K] alpha_hat_scale;// sd's of the means of predictors
  real<lower = -1, upper = 1> corr_alpha;////vector<lower = -1, upper = 1>[[(K*(K-1))/2]] corr_alpha;
  
  
  // individual-level parameters
  matrix[I,K*K] z_beta;
  matrix[I,K] z_alpha;
  matrix[I,K] z_tau;
  real corr[I];
  
  corr_matrix[K*K] Omega_beta;
  
  vector[N_mis[1]] y_mis1;
  vector[N_mis[2]] y_mis2;
}

transformed parameters {
  // want to add exponent of tau and draw tau's in same way as alpha's and beta's, from multivariate distribution
  matrix[K, K] betas[I]; // individual VAR(1) coefficient matrix
  vector[K] alpha[I]; // individual means matrix
  matrix[I, K] tau_int; // individual sd's matrix, normal distribution
  matrix[I, K] tau; // individual sd's matrix, log-normal distribution
  matrix[K,K] Omega[I];
  matrix[K,K] Omega_alpha;
  matrix[K,K] Omega_tau;
  
  matrix[K, K] L_alpha;
  matrix[(K*K), (K*K)] L_beta;
  matrix[K, K] L_tau;
  
  vector[K] Y[N]; // Dependent variables
  vector[K] X[N];
  
  vector[K] Y_std[N]; 
  vector[K] X_std[N];
  
  // Transformations to get data into shards
  // 200 shards. Run per individual and combine
  // M = N/200 = 14000/200 = 70
  //int n_shards = I;
  //int M = T;// N/n_shards;
  //int xi[n_shards,M];  
 
 
  Omega_alpha = diag_matrix(rep_vector(1,K));
  Omega_alpha[1,2] = corr_alpha;
  Omega_alpha[2,1] = corr_alpha;
  
  Omega_tau = diag_matrix(rep_vector(1,K));
  Omega_tau[1,2] = corr_tau;
  Omega_tau[2,1] = corr_tau;
    
  L_alpha = cholesky_decompose(quad_form_diag(Omega_alpha, alpha_hat_scale));
  L_beta = cholesky_decompose(quad_form_diag(Omega_beta, beta_hat_scale)); // some rejections occurs here
  L_tau = cholesky_decompose(quad_form_diag(Omega_tau, tau_hat_scale));
  
  for(i in 1:I) { // Can't remove this loop, because I need I alpha and beta's in output
    alpha[i] = alpha_hat_location + L_alpha * z_alpha[i]';
    // implies: alpha[i] ~ multi_normal(alpha_hat_location, alpha_hat_scale)
    betas[i] = to_matrix((beta_hat_location + L_beta * z_beta[i]'), K, K);
    
    tau_int[i] = to_row_vector(tau_hat_location + L_tau * z_tau[i]');
    // Transform tau to make lognormal
    tau[i] = exp(tau_int[i]);
    
    Omega[i] = diag_matrix(rep_vector(1,K));
    Omega[i,1,2] = corr[i];
    Omega[i,2,1] = corr[i];
    }
    
  
  //Y = Yobs;
  //X = Xobs;
  Y[comp_ind1,1] = to_array_1d(Ycomp1);
  Y[comp_ind2,2] = to_array_1d(Ycomp2);
  
  X[comp_ind1_x,1] = to_array_1d(Xcomp1);
  X[comp_ind2_x,2] = to_array_1d(Xcomp2);
  //X[comp_ind1_x,1] = to_array_1d(Xcomp1);
  //X[comp_ind2_x,2] = to_array_1d(Xcomp2);
  
  Y[miss_ind1,1] = to_array_1d(y_mis1);
  Y[miss_ind2,2] = to_array_1d(y_mis2);
  
  X[miss_ind1_x,1] = to_array_1d(y_mis1[miss_ind1_xm1]);
  X[miss_ind2_x,2] = to_array_1d(y_mis2[miss_ind2_xm1]);
  
  
  Y_std[,1] = to_array_1d((to_vector(Y[,1]) - meanY[1])/sdY[1]);
  Y_std[,2] = to_array_1d((to_vector(Y[,2]) - meanY[2])/sdY[2]);
  
  X_std[,1] = to_array_1d((to_vector(X[,1]) - meanX[1])/sdX[1]);
  X_std[,2] = to_array_1d((to_vector(X[,2]) - meanX[2])/sdX[2]);

}

// The model to be estimated.
model {
  // hyperpriors
  // To get mean of 3 and var of 1 with lognormal set normal mean to -0.111572 and sd to 0.4723812
  // for mean(var) inno's of 3(1), 5(1), 5(2), and 7(3), normal mean and sd vary between 1 and 2, and .04 and .10
  // respectively
  tau_hat_location ~ normal(0, 5) ; // average SD between -2 and 2, scaled so lognormal comes out right
  tau_hat_scale ~ normal(0, 0.5); // SD of SD between 0 and 1 (half-normal)
  alpha_hat_location ~ normal(0, 5); // average mean between 0 and 6, seems reasonable given most likert-scales used 
  // in social science
  alpha_hat_scale ~ normal(0, 1); // SD of means (SE's) between 0 and 4 (half-normal)
  beta_hat_location ~ normal(0, 0.5);// AR and cross-lagged parameters between -.5 and .5
  beta_hat_scale ~ normal(0.5, 0.5);// SE's of AR- and cross-lagged between 0 and .6 (half-normal)
  c_location ~ normal(0, 0.7);// Correlation between residuals between -7 and .7
  c_scale ~ normal(0.5, 0.5); // SE's of correlations between 0 and .6 (half-normal)
  corr_alpha ~ normal(0, 0.7); // correlation between means between -.7 and .7
  corr_tau ~ normal(0, 0.7); // correlation between sd's of DV's between -.7 and .7
  
  Omega_beta ~ lkj_corr(2);//2
  
  // hierarchical priors
  // non-centered parameterization
  to_vector(z_alpha) ~ std_normal();
  to_vector(z_tau) ~ std_normal();
  to_vector(z_beta) ~ std_normal();
  y_mis1  ~ normal(meanY[1], (2*sdY[1]));
  y_mis2  ~ normal(meanY[2], (2*sdY[2]));
  //y_mis1  ~ normal(4, 4);
  //y_mis2  ~ normal(4, 4);
 
 
   for ( j in 1:I ) {
    
    corr[j] ~ normal(c_location, c_scale); 
  
   }
   
   
  
  {
  
  vector[1] beta;
  vector[(10*M)+6] theta[n_shards]; // 6M + 8 because two means, two AR-prs, 2 cl-pars, Omega and tau, 6*M if Omega and tau fixed
 
  
  for ( i in 1:n_shards ) {
    int j = 1 + (i-1)*M;
    int k = i*M;
    
    theta[i,1:M] = to_vector(Y_std[j:k, 1 ]);
    theta[i,(M+1):(2*M)] = to_vector(Y_std[j:k, 2 ]);
    theta[i,((2*M)+1):(3*M)] = to_vector(X_std[j:k, 1 ]);
    theta[i,((3*M)+1):(4*M)] = to_vector(X_std[j:k, 2 ]);
    
    theta[i,((4*M)+1):(5*M)] = to_vector(alpha[individual[j:k],1]);
    theta[i,((5*M)+1):(6*M)] = to_vector(alpha[individual[j:k],2]);
    theta[i,((6*M)+1):(7*M)] = to_vector(betas[individual[j:k],1,1]);
    theta[i,((7*M)+1):(8*M)] = to_vector(betas[individual[j:k],K,K]);
    theta[i,((8*M)+1):(9*M)] = to_vector(betas[individual[j:k],1,K]);
    theta[i,((9*M)+1):(10*M)] = to_vector(betas[individual[j:k],K,1]);
    theta[i,((10*M)+1):(((10*M)+1)+3)] = to_vector(Omega[individual[j]]);
    theta[i,((10*M)+5):(((10*M)+5)+1)] = to_vector(tau[individual[j],1:K]);
    
    
  }
  
  
  beta[1] = T; // Number of observations per shard
  
   target += sum(map_rect(lp_reduce, beta, theta, xr, xi));
  
  }

}

generated quantities {
  matrix[K, K] betas_raw[I]; // individual VAR(1) coefficient matrix
  vector[K] alphas_raw[I]; // individual means matrix
  matrix[I, K] tau_raw; // individual sd's matrix, log-normal distribution
  
  vector[K] alpha_hat_location_raw;
  vector[K] alpha_hat_scale_raw; 
  matrix[K, K] beta_hat_location_raw;
  matrix[K, K] beta_hat_scale_raw;
  
  vector[K] tau_hat_location_raw ; 
  vector[K] tau_hat_scale_raw ; 
  vector[K] tau_hat_mode_raw ; 
  
  real corr_tau_raw;
  
  //real y_mis1_raw[N_mis[1]];
  //real y_mis2_raw[N_mis[2]];
  
  //y_mis1_raw =  to_array_1d((sdY[1] * to_vector(y_mis1))) + meanY[1];
  //y_mis2_raw =  to_array_1d((sdY[2] * to_vector(y_mis2))) + meanY[2];
  
  //y_mis1_raw =  to_array_1d((sdY[1] * to_vector(y_mis1)) + meanY[1]);
  //y_mis2_raw =  to_array_1d((sdY[2] * to_vector(y_mis2)) + meanY[2]);
  
  alphas_raw[,1] = to_array_1d((sdY[1] * (to_vector(alpha[,1]))) + meanY[1]);
  alphas_raw[,2] = to_array_1d((sdY[2] * (to_vector(alpha[,2]))) + meanY[2]);
  
  betas_raw[,1,1] = to_array_1d(to_vector(betas[,1,1]) * sdY[1]/sdX[1]);
  betas_raw[,1,2] = to_array_1d(to_vector(betas[,1,2]) * sdY[1]/sdX[2]);
  betas_raw[,2,1] = to_array_1d(to_vector(betas[,2,1]) * sdY[2]/sdX[1]);
  betas_raw[,2,2] = to_array_1d(to_vector(betas[,2,2]) * sdY[2]/sdX[2]);
  
  tau_raw[,1] = exp(tau_int[,1])*sdY[1];
  tau_raw[,2] = exp(tau_int[,2])*sdY[2];
  
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
  
  tau_hat_location_raw[1] = exp(tau_hat_location[1] + ((tau_hat_scale[1]^2)/2))*sdY[1];
  tau_hat_location_raw[2] = exp(tau_hat_location[2] + ((tau_hat_scale[2]^2)/2))*sdY[2];
  
  tau_hat_mode_raw[1] = exp(tau_hat_location[1] - (tau_hat_scale[1]^2))*sdY[1];
  tau_hat_mode_raw[2] = exp(tau_hat_location[2] - (tau_hat_scale[2]^2))*sdY[2];

  tau_hat_scale_raw[1] = sqrt((exp((tau_hat_scale[1])^2)-1)*exp((2*tau_hat_location[1]) + ((tau_hat_scale[1])^2)))*sdY[1];
  tau_hat_scale_raw[2] = sqrt((exp((tau_hat_scale[2])^2)-1)*exp((2*tau_hat_location[2]) + ((tau_hat_scale[2])^2)))*sdY[2];
  
  
  // Equation to turn correlation between normal vars into correlation between their exp counterparts. 
  corr_tau_raw = (
                 (
                 (exp(tau_hat_location[1] + tau_hat_location[2])) * 
                 (exp((((tau_hat_scale[1])^2)/2) + (((tau_hat_scale[2])^2)/2) + corr_tau))
                 ) - 
                 (
                 (exp(tau_hat_location[1] + tau_hat_location[2])) * 
                 (exp((((tau_hat_scale[1])^2)/2) + (((tau_hat_scale[2])^2)/2)))
                 )
                 )
                 /
                 (
                 sqrt(
                 (exp((tau_hat_scale[1])^2)-1)*
                 exp(2*tau_hat_location[1] + ((tau_hat_scale[1])^2))
                 )*
                 sqrt(
                 (exp((tau_hat_scale[2])^2)-1)*
                 exp(2*tau_hat_location[2] + ((tau_hat_scale[2])^2))
                 )
                 );
  
}

