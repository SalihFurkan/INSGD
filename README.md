# INSGD

The paper is available in https://openreview.net/pdf?id=5TaBxctwRZ.

In this paper, we propose a novel optimization algorithm for training machine learning models called Input Normalized Stochastic Gradient Descent (INSGD), inspired by the Normalized Least Mean Squares (NLMS) algorithm used in adaptive filtering. When training complex models on large datasets, choosing optimizer parameters, particularly the learning rate, is crucial to avoid divergence. Our algorithm updates the network weights using stochastic gradient descent with $\ell_1$ and $\ell_2$-based normalizations applied to the learning rate, similar to NLMS. However, unlike existing normalization methods, we exclude the error term from the normalization process and instead normalize the update term using the input vector to the neuron. Our experiments demonstrate that our optimization algorithm achieves higher accuracy levels compared to different initialization settings. We evaluate the efficiency of our training algorithm on benchmark datasets using a toy neural network and several mature modern deep networks including ResNet-20, ResNet-50, MobileNetV3, WResNet-18, and Vision Transformer.
% BERT, 
Our INSGD algorithm improves ResNet-20's CIFAR-10 test accuracy from 92.57\% to 92.67\%, MobileNetV3's CIFAR-10 test accuracy from 90.83\% to 91.13\%, WResNet-18 on CIFAR-100 from 78.24\% to 78.47\%, and ResNet-50's accuracy on ImageNet-1K validation dataset from 75.60\% to 75.92\%. 

Let us assume that we have a linear neuron at the last stage of the network. 
let $e_n$ be the error due to the n-th training instance.
We update the weight vector $w_k$ of the k-th neuron of the last layer as follows

$w_{k+1} = w_k - \lambda \frac{e_n}{||x_n||^2_2}    x_n $

where $x_n$ is the input feature map vector of the k-th neuron due to the
$n$-th instance. This equation is essentially the same as the NLMS equation.

![alg_fig](https://github.com/SalihFurkan/Normalized-SGD/assets/54649498/ef410a25-0877-4391-8f08-b3f08fa5584c)

The figure showing how the optimizer algorithm works for any layer is shown in the figure. For any other layer beside the last layer, the formula becomes slightly different. We use the gradient term as it is and normalize it with the input. 

$w_{k+1} = w_k - \mu \frac{\nabla_{w_k}L(e_k)}{\epsilon + ||x_k||^2_2}$

We incorporate momentum, a technique that aids in navigating high error and low curvature regions. In the INSGD algorithm, we introduce an input momentum term to estimate the power of the dataset, enabling power normalization. By replacing the denominator term with the estimated input power, we emphasize the significance of power estimation in our algorithm. Furthermore, the utilization of input momentum allows us to capture the norm of all the inputs. Denoted as $P$, the input momentum term accumulates the squared $\ell_2$ norm of the input instances:

$P_k = \beta P_{k-1} + (1-\beta)||x_{k}||_2^2.$

While estimating the input power is crucial, we encounter a challenge similar to AdaGrad. The normalization factor can grow excessively, resulting in infinitesimally small updates. To address this, we employ the logarithm function to stabilize the normalization factor. However, the use of the logarithm function introduces the risk of negative values. If the power is too low, the function could yield a negative value, reversing the direction of the update. To mitigate this, we employ a function with the rectified linear unit, which avoids the issue of negative values. Adding a regularizer may not be sufficient to resolve this problem, hence the choice of the rectified linear unit function. The update equation:

$w_{k+1} = w_k - \frac{\mu}{f_{\epsilon}(log(P_k))} \nabla_{w_k}L(\mathbf{e}_k)$
