# STORM(STOchastic-Recursive-Momentum)
#Objective
Variance reduction has emerged in recent years as a strong competitor to stochastic gradient descent in non-convex problems, providing the first algorithms to improve upon the converge rate of stochastic gradient descent for finding first-order critical points. However, variance reduction techniques typically require carefully tuned learning rates and willingness to use excessively large “mega-batches" in order to achieve their improved results.
In this project, we implement the algorithm, STORM, that does not require any batches and makes use of adaptive learning rates, enabling simpler implementation and less hyperparameter tuning.  This algorithm was first implemented in the paper “Momentum- Based Variance reduction in non-Convex SGD” by Ashok Cutkosky and Francesco Orabona. 

