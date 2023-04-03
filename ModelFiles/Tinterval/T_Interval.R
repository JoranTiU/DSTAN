# data: a vector containing the variable on which you want to insert phantom missings
# ID: a vector of the same length as data which indicates which participant each line belongs to.
# Time: a vector of the same length as data which indicates which measurement occasion each line belongs to.
# The function returns a dataframe with phantom missings introduced for all participants on the outcome variable. 
# The dataframe contains the outcome variable, an indicator for which participant a row belings too, and an indicator
# which measurement occasion a row belongs too.

t_interval <- function(data, ID = ID, Time = Time){
  
  lower <- 1
  upper <- max(Time)
  
  N_indiv <- length(unique(ID))
  
  ID_restr <- rep(unique(ID), each = upper)
  Time_restr <- rep(lower:upper, N_indiv)
  
  data_restr <- as.data.frame(matrix(NA, nrow = N_indiv*upper, ncol=1))
  colnames(data_restr) <- c("outcome")
  
  data_restr$ID <- ID_restr
  data_restr$Time <- Time_restr
 
  
  for(i in 1:length(ID_restr)){
    
    ind <- ID_restr[i]
    ind2 <- which(unique(ID) == ind)
    
    for(j in 1:upper){
      
      if(j %in% Time[which(ID == ind)]){
        
          data_restr[(((ind2-1)*upper)) + j, 1] <- data[which(ID == ind)][match(j, Time[which(ID == ind)])]
          
      }
      
      data_restr
      j <- j + 1
      
    }
  }
  
  return(data_restr)
}

# example use: for outcome variable "PosAffect", ID variable "ID_samp", and Time variable "Time_samp"

# PosAffect_phantom <- t_interval(PosAffect, ID_samp, Time_L)