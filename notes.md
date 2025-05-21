## Running with PowerLaw+NegBinomial

Starting with weiner filter really helps. 




### R = 3.906

- **N=64**
It seems that 250 steps the chain has converged EXCEPT for the largest scale mode. bias params are ok. With non-dense mass matrix.  
For a dense mass matrix the thing does not seem to work.

I tried with 1LPT and MND=1e-2 and still fails to reconstruct large scale.
Only seems to work with R=8 (ALPT and so on). 

I tried now with MND=5e-2. cweb_sharpness = 7. Arund 12 min expected. 


- N=128
The smallest k-mode, ie largest scale, does nto converge properly. bad P(k) ratio. I do not know if it is ALPT, the radial RSD, the mean number density, or what. 

### R = 8 Mpc/h.
- N=64. Works well. I could use 2LPT bc ALPT not needed at those scales. Good convergence with 250 mcmc steps. let's do that. With N=128 it is 1 Gpc which is good.  


### R = 5 Mpc/h.
- N=64. Works well. Total convergence with 250 steps. Full model. 




