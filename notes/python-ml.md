# MACHINE LEARNING NOTES

## DATA PREPROCESSING 
-import the data 
- clean the date 
- split into training and test sets 

## MODELLING 
- build the model
- train the model 
- make predictions 

## EVALUATIONS
- calculate preformance metrics
- make verdict 

## FEATURE SCALING 
- applied to the columns in datasets 
- normalization 
- standardization 

-----------------------------------------------------------

## ASSUMPTIONS OF LINEAR REGRESSIONS 

# Linearity 
- X and Y have a linear relation 
# Homoscedasticity
- equal variance 
# Multivariate Normality 
- normality of error distribution 
# Independence 
- of observations which includes no autocorrelation
# Lack of Multicollinearity 
- Predictors are not correlated with each other 
# The Outlier Check 
- This is not an assumption but an extra

-----------------------------------------------------------

## STATISTICAL SIGNIFICANCE 

# P VALUES 
- usually significance level at 5% or 0.05 but can be adjustable 
- when values go below the significance level then we can reject the original hypothesis 

-----------------------------------------------------------

## BUILDING A MODEL (MULTILINEAR REGRESSIONS)

# All in 
- prior knowledge 
- you have to 
- preparing for backward elimination 
# Backward Elimination 
- Select a significance level to stay in the model (SL=0.05)
- Fit the full model with all the possible predictors 
- Consider the predictor with the highest P value. If P>SL then go to step four or else FIN (model is ready)
- Remove the predictor 
- Fit the model without this variable 
- Return to step c and repeat 
# Forward Selection 
- Select a significance level to enter model (SL=0.05)
- Fit all simple regression models y~xn, Select the one with the lowest P value 
- Keep this variable and fit all possible models with one extra predictor added to the ones you already have
- Consider the predictor with the lowest P value. If P<SL then go back to last step or else FIN (Keep the previous model)
# Bidirectional Elimination
- Select a significance level to enter and to stay in the model (SLENTER =0.05 OR SLSTAY=0.05)
- Perform the next step of Forward selection (new variables must have P < SLENTER to enter)
- No new variables can enter and no old variables can exit 
- FIN or model is ready  
# All Possible Models 
- Select a criterion of goodness of fit (Akaike criterion) 
- Construct all possible regression models 2^n-1 total combinations 
- Select the one with the best criterion 
- FIN = model is ready