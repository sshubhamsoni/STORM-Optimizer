# STORM(STOchastic-Recursive-Momentum)

## Objective

Variance reduction has emerged in recent years as a strong competitor to stochastic gradient descent in non-convex problems, providing the first algorithms to improve upon the converge rate of stochastic gradient descent for finding first-order critical points. However, variance reduction techniques typically require carefully tuned learning rates and willingness to use excessively large “mega-batches" in order to achieve their improved results.
In this project, we implement the algorithm, STORM, that does not require any batches and makes use of adaptive learning rates, enabling simpler implementation and less hyperparameter tuning.  This algorithm was first implemented in the paper “Momentum- Based Variance reduction in non-Convex SGD” by Ashok Cutkosky and Francesco Orabona. 


### Momentum and Variance Reduction

The stochastic gradient descent with momentum is typically implemented as 
![](Images/SGD%20with%20momentum.png)
Where a is small, a = 0.1. In other words, instead of using the current gradient ∇F(X_t) in the update step ofx_t, we use an exponential average of the past observed gradients.
	
While SGD with momentum and its variants have been successfully used in many machine learning applications, it is well known that the presence of noise in the stochastic gradients can nullify the theoretical gain of the momentum term. As a result, it is unclear how and why using momentum can be better than plain SGD. Although recent works have proved that a variant of SGD with momentum improves the non-dominant terms in the convergence rate of convex stochastic least square problems, it is still unclear if the actual convergence rate can be improved.
Here, we take a different route. Instead of showing that the momentum in SGD works in the same way as in noiseless case, i.e. giving accelerated rates, we show that a variant of momentum can probably reduce the variance of the gradients. In its simplest form, the variant we propose is:
![](Images/SGD%20with%20a%20variant%20of%20momentum.png)

The only difference is that we add the term (1-α)(∇f(x_t,ϵ_t )  - ∇f(x_(t-1),ϵ_t )) to the update. As in standard variance-reduced methods, we use two gradients in each step. However, we do not need to use the gradient calculated at any checkpoints. Note that if  x_t 〖≈x〗_(t-1) , then our update becomes approximately the momentum one. These two terms will be similar as long as the algorithm is actually converging to some point, and so we can expect the algorithm to behave exactly like the classic momentum SGD towards the end of the optimization process.
![](Images/STORM%20algorithm.png)

### Empirical Validation

In order to confirm that the algorithm performs well and require little tuning, we implemented STORM in PyTorch and tested its performance on the CIFAR-10 image recognition benchmark using a ResNet-32 Model.  We compare STORM to AdaGrad and Adam, which are both very popular and successful optimization algorithms. The learning rates for AdaGrad and Adam were swept over a logarithmically spaced grid. For STORM, we set ω=k=0.1 as default and swept c over a logarithmically spaced grid.
We record train loss (cross entropy), and accuracy on both the train and test sets.
These results show that, while STORM is only marginally better than AdaGrad on test accuracy, on both training loss and accuracy STORM appears to be somewhat faster in terms of number of iterations. We note that the convergence proof we provide actually only applies to the training loss (since we are making multiple passes over the dataset).   

## Conclusion

We implemented the variance-reduction-based algorithm, STORM, that finds critical points in stochastic, smooth, non-convex problems. Our algorithm improves upon prior algorithms by virtue of removing the need for checkpoint gradients, and incorporating adaptive learning rates. These improvements mean that STORM is substantially easier to tune: it does not require choosing the size of checkpoints, nor how often to compute the checkpoints (because there are no checkpoints), and by using adaptive learning rates, the algorithm enjoys the same robustness to learning rate tuning as popular algorithms like AdaGrad or Adam. STORM obtains the optimal convergence guarantee, adapting to the level of noise in the problem without knowledge of this parameter. We verified that on CIFAR-10 with a ResNet-32 Architecture, STORM indeed seems to be optimizing the objective in fewer iterations than baseline algorithms.
	Additionally, we point out that STORM’s update formula is strikingly similar to the standard SGD with momentum heuristic employed in practice.  To our knowledge, no theoretical result actually establishes an advantage of adding momentum to SGD in stochastic problems, creating an intriguing mystery. While our algorithm is not precisely the same as the SGD with momentum, we feel that it provides strong intuitive evidence that momentum is performing some kind of variance reduction.   

