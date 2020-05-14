# Option-Pricing-Using-Python
Methods
1. Analytical methods / Closed form solutions ==> Black-Scholes Model 
2. Numerical methods 
The value of a derivate, in the vast majority of situations, has no analytical solution, and we turn to numerical approach. 
  - Finite Different Methods
  prons:
  can be adapted to a wide variety of UA price dynamics. 
  well suited to price American options
  cons:
  instability ==> horizontal, rectangular shape; finer space discretiztion, even finer time discretization 
  curse of dimensionality ==> difficut to implement as the number of state variables increases 
  
    - Explicit Finite Differences Methods(EFDM) -- recursive algorithm
    create a time-space-discreted grid, approximation of Greeks, boudary conditions 
    for American option, systematically take the maximum between the exercise value and the continuation value 
    - Implicit Finite Differences Methods(IFDM)
    - Crank-Nicolson algorithm 
  curse of dimensionality ==> difficut to implement as the number of state variables increases 
  
  - Monte Carlo Simulation 
  simulating paths for the price of UA
  prons:
  flexible, the difficult does not increase with dimension
  cons:
  challenging to evaluate American options 
    Monte Carlo Simulation + Dynamic pricing 
    find the optimal exercise policies - trial and error

    - Numerical integration 
    - Variation reduction techniques 
    - Conditional Monte Carlo Simulation 
   
   - Dynamic programmming  
    


Discrete time approach 
- the binomial tree + convergence improvement 
- the trinomial tree 
- managing imperfect replication 


Pricing exotic option 
- Barrier type options 
  - Vanilla barrier option
  - Parisian options 
- Average-type options
  - Asian option with geometric averaging 
  - Asian option with arithmetic averaging 
- Lookback options 
- Spread options 
- Compound options 
- Bermudian options 

Underlying asset dynamics 



Finite 
