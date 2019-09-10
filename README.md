### Abstract

The experiments investigate the performace of **stochastic prox-linear** based methods compared to the prototypical
stochastic gradient methods on common optimization problems from supervised machine learning. We focus on solving
problems with nonsmooth loss functions and nonlinear prediction functions. More precisely, the methods solve problems
of the form

```
min L(h(x)) + r(x)
```

where ```L``` is nonsmooth convex, ```h``` is a smooth nonlinear operator and ```r``` is possibly nonsmooth nonconvex.
We consider stochastic optimization problems, thus the convex composite ```L(h(x))``` is an expectation.

The code contains the implementation of the used methods as well as the experiments on different optimization problems.

### Requirements

The implementation uses python 3.7 and the following libraries.

```
numpy
pytorch
click
evaltool (https://github.com/JeGa/evaltool)
```

### Problems

We provide implementations for the following problems.

- **Logistic regression** with a fully connected neural network as prediction function on an artificially
    generated data set.
- **Robust nonlinear regression** with an exponential prediction function on an artificially generated data set.
- **Multiclass classification** with SVM/Hinge loss and a convolutional neural network as prediction function using
    the MNIST data set.

The folder ```scripts``` contains various scripts for generating plots and examples.

### References

- Adrian S Lewis and Stephen J Wright. “A proximal method for composite minimization”.
- Dmitriy Drusvyatskiy and Adrian S Lewis. “Error bounds, quadratic growth, and linear convergence of proximal methods”.
- Damek Davis and Dmitriy Drusvyatskiy. “Stochastic model-based minimization of weakly convex functions”.
- MNIST handwritten digit database, Yann LeCun, Corinna Cortes and Chris Burges.
    June 2019. url: http://yann.lecun.com/exdb/mnist/.
