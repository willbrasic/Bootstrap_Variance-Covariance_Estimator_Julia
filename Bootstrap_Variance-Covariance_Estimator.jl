""" 
The purpose of this program is to simulate data and construct the
bootstrap and wild bootstrap variance-covariance matrix. Then,
I construct the empirical T and empirical Wald statistic distributions
to conduct hypothesis tests with the bootstrap and wild bootstrap p-values.

"""

__author__ = "William Brasic"
__email__ =  "wbrasic@arizona.edu"


"""

Preliminaries

"""

# importing necessary libraries
using Random; using Distributions; using Statistics; using DataFrames; using GLM; using StatsBase


"""

Data simulation

"""


# sample size
n = 1000;

# number of bootstraps
B = 10000;

# significance level α
α = 0.05;

# number of covariates
k = 7;

# setting seed for reproducibility
Random.seed!(1024);

# dataframe
df = DataFrame(x2 = rand(Binomial(1, 0.5), n), x3 = rand(Binomial(1, 0.2), n));  

#creating x1
#For each x ∈ df.x2, call it x, if x == 1 then use certain distribution;
# if x ̸∈ df.x2, use a different distribution.
#Then, generate a new column based on this data called x1.
transform!(df, :x2 => (x -> ifelse.(x .== 1, 
                                    sample([10, 12, 14, 16, 18, 20], 
                                    Weights([0.1, 0.3, 0.2, 0.35, 0.03, 0.02]), 
                                    n, replace = true) , 
                                    sample([10, 12, 14, 16, 18, 20], 
                                    Weights([0.05, 0.3, 0.15, 0.42, 0.06, 0.02]), 
                                    n, replace = true))) => :x1)
 

# for each x ∈ df.x1 create a normal random variable with variance based on x^2.
df.e = reduce(vcat, map(x -> rand(Normal(0,  sqrt(225 / x^2)), 1), df.x1));

# creating y 
df.y = 1 .+ 0.13 .* df.x1 .- 0.03 .* df.x2 .- 0.02 .* df.x1 .* df.x2 .- 0.02 .* df.x3 .+ df.e;

# move x1 to first column in place
select!(df, :y, :x1, :);


"""

Bootstrap

"""


# setting seed for reproducibility
Random.seed!(1024);

# running OLS on initial sample  
model = lm(@formula(y ~ x1 + x2 + x3 + x1*x2 + x1*x3 + x2*x3 + x1*x2*x3), df);
β_hat = coef(model);
se_β_hat = stderror(model);

# regressor matrix
X = hcat(repeat([1], n), df.x1, df.x2, df.x3, df.x1.*df.x2, 
         df.x1.*df.x3, df.x2.*df.x3, df.x1.*df.x2.*df.x3);

# sum of residuals squared scaled by degrees of freedom correction
σ_hat = sum(residuals(model).^2)/(n - k - 1);

# initializing matrix for t statistics and vector for wald statistics
T_pt = zeros(Float64, B, k + 1); Wald = zeros(B);

# number of restrictions for wald test
r = 4;

# c vector and C matrix for Wald test of β_4 = β_5 = β_6 = β_7 = 0
c = [0, 0, 0, 0]; C = [0 0 0 0 1 0 0 0
                       0 0 0 0 0 1 0 0
                       0 0 0 0 0 0 1 0
                       0 0 0 0 0 0 0 1 ];

# wald statistic using normal OLS var-cov matrix 
W = (C*β_hat - c)' * inv(σ_hat * C * inv(X' * X) * C' ) * (C*β_hat - c);

@doc """
       This function estimates the empirical distribuiton for T and Wald statistics using Bootstrap
       """ ->
function hypo_test_stats_boot(data, T_empty_matrix, Wald_empty_vector, 
                              β_hat = β_hat, n = n, k = k, c = c, C = C)
    for i in 1:B

        # resampling the dataframe with replacement
        df1 = data[sample(1:n, n, replace = true), :]
        
        # getting y from resampled dataframe
        y = Vector(df1[:, :y])
    
        # creating regressor matrix with summer vector
        X = hcat(repeat([1], n), Matrix(df1[:, [:x1, :x2, :x3]]), 
                df1.x1 .* df1.x2, df1.x1 .* df1.x3, df1.x2 .* df1.x3, 
                df1.x1 .* df1.x2 .* df1.x3)
    
        # estimates of β
        boot_β_hat = inv(X' * X) * X' * y 
    
        # residual estimates 
        boot_e_hat = y .- X * boot_β_hat;

        # error variance estimate
        σ_hat = sum(boot_e_hat.^2)/(n - k - 1);
    
        # variance covariance matrix 
        vc_matrix = inv(X' * X) * σ_hat;
    
        # standard error
        se_boot_β_hat = [sqrt(vc_matrix[i,i]) for i ∈ 1:(k+1)];
    
        # T statistic and storing it
        T_empty_matrix[i,:] = (boot_β_hat .- β_hat)./(se_boot_β_hat)

        # Wald statistic and storing it
        Wald_empty_vector[i] = (C*boot_β_hat - c)' * inv(σ_hat * C * inv(X' * X) * C' ) * (C*boot_β_hat - c);
    
    end
end;

# running the created function
hypo_test_stats_boot(df, T_pt, Wald)

# bootstrap p-value to test that the coefficient on X_2 (β_2) is zero
( 1/B ) * ( sum( abs.(T_pt[:, 3]) .> abs( β_hat[3] / se_β_hat[3] ) ) )

# bootstrap p-value to test that the coefficient on X_1*X_2 (β_4) is zero
( 1/B ) * ( sum( abs.(T_pt[:, 5]) .> abs( β_hat[5] / se_β_hat[5] ) ) )

# bootstrap p-value to test if β_4 = β_5 = β_6 = β_7 = 0.
( 1/B ) * ( sum( Wald .> W ) )

"""

Notice that the bootstrap p-value on the coefficient for x2 is 0.8551. 
Hence, fail to reject the null that β_hat_2 = 0.

Notice that the bootstrap p-value on the coefficient for x1*x2 is 0.363. 
Hence, fail to reject the null that β_hat_4 = 0.

Notice that the bootstrap p-value on the coefficient for the Wald test statistic is 0.9308.
Hence, we fail to reject the null that β_4 = β_5 = β_6 = β_7 = 0.

"""


# setting seed for reproducibility
Random.seed!(1024);

# running OLS on initial sample  
model = lm(@formula(y ~ x1 + x2 + x3 + x1*x2 + x1*x3 + x2*x3 + x1*x2*x3), df);
β_hat = coef(model); se_β_hat = stderror(model);

# sum of residuals squared scaled by degrees of freedom correction
σ_hat = sum(residuals(model).^2)/(n-k-1)

# regressor matrix
X = hcat(repeat([1], n), df.x1, df.x2, df.x3, df.x1.*df.x2, 
         df.x1.*df.x3, df.x2.*df.x3, df.x1.*df.x2.*df.x3);

# initializing matrix for t statistics and vector for wald statistics
T_pt = zeros(Float64, B, k + 1); Wald = zeros(B);


@doc """
       This function estimates the empirical distribuiton for T and Wald statistics using Wild Bootstrap
       """ ->
function hypo_test_stats_wild(T_empty_matrix, Wald_empty_vector, X = X, σ_hat = σ_hat, 
                              β_hat = β_hat, n = n, k = k, c = c, C = C)
    for i in 1:B

        # creating y per Liu (1998)
        y = X * β_hat + σ_hat .* sample([1, -1], Weights([0.5, 0.5]), n, replace = true)  

        # estimates of β
        wild_β_hat = inv(X' * X) * X' * y 
    
        # residual estimates 
        wild_e_hat = y .- X * wild_β_hat

        # error variance estimate
        wild_σ_hat = sum(wild_e_hat.^2)/(n - k - 1);
    
        # variance covariance matrix 
        wild_vc_matrix = inv(X' * X) * wild_σ_hat;
    
        # standard error
        se_wild_β_hat = [sqrt(wild_vc_matrix[i,i]) for i ∈ 1:(k+1)];
    
        # T statistic and storing it
        T_empty_matrix[i,:] = (wild_β_hat .- β_hat)./(se_wild_β_hat)

        # Wald statistic and storing it
        Wald_empty_vector[i] = (C*wild_β_hat - c)' * inv(wild_σ_hat * C * inv(X' * X) * C' ) * (C*wild_β_hat - c);
    
    end
end;

# calling created function
hypo_test_stats_wild(T_pt, Wald);

# wild bootstrap p-value to test that the coefficient on X_2 (β_2) is zero
( 1/B ) * ( sum( abs.(T_pt[:, 3]) .> abs( β_hat[3] / se_β_hat[3] ) ) )

# wild bootstrap p-value to test that the coefficient on X_1*X_2 (β_4) is zero
( 1/B ) * ( sum( abs.(T_pt[:, 5]) .> abs( β_hat[5] / se_β_hat[5] ) ) )

# wild bootstrap p-value to test if β_4 = β_5 = β_6 = β_7 = 0.
( 1/B ) * ( sum( Wald .> W ) )


"""

Notice that the wild bootrap p-value on the coefficient for x2 (education) is 0.848. 
Hence, fail to reject the null that β_hat_2 = 0.

Notice that the wild bootstrap p-value on the coefficient for x21*x2 is 0.363. 
Hence, fail to reject the null that β_hat_4 = 0.

Notice that the wild bootstrap p-value on the coefficient for the Wald test statistic is 0.9218.
Hence, we fail to reject the null that β_4 = β_5 = β_6 = β_7 = 0.

"""