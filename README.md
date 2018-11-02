## Efficient LR Machine Learning End-to-End Distributed Training on Spark.
* Author: Tianbing XU
* Code Built: Nov. 2015.

### Core Methodologies
* Problem: Image Search Ads CTR (click-through-rate) prediction with queries/clicks and images.
* Adopted Distributed Conjugate Gradient algorithm on Spark Distributed Computing Framework.
* Developped fast and accuracte End-to-End Logistic Regression Training Pipelines from raw data to model to prediction.

### Performance (Scalability and Accuracy)
* A highly scalable Machine Learning training system deployed on 100+ machines for 10+ TeraBytes data.  
* 10X-20X speedup with much more better test accuracy compared to MLLib SGD
* Achieved near production AUC accuracy (on real online traffic testing) with much less number of machines.

## Distributed Computing Architecture
### Communication vs Computation Tradeoff
The motivation behind introducing full batched Conjugate Gradient to distributed workers is to achieve better 
tradeoff between communication and computation. As communication cost (one or two orders slower than in-memory computation) 
becomes the bottleneck of the training time, it is highly desirable to reduce the communication rounds from ![equation](http://latex.codecogs.com/gif.latex?O%281/%5Cepsilon%29%24%20%28SGD%29) to ![equation](http://latex.codecogs.com/gif.latex?log%281/%5Cepsilon%29). Thus, Conjugate Gradient becomes an excellent choice. L-BFGS is another choice with extra 
memory cost for really large training data. Here, we observed Out-of-Memory issues for MLLib L-BFGS implementation, 
it really depends on a good implementation. As for LR, since the Hessian matrix has a simple,
closed, diagonal form, it is at the same effiency to transfer the Hessian matrix (or ![equation](http://latex.codecogs.com/gif.latex?\mu%27%20H%20\mu)) as a gradient vector between the master and worker nodes via network.

### Architecture Design
We implement Parameter Server Architecture based on the Spark Distributed Computing Framework.
![archicture](https://user-images.githubusercontent.com/3246048/45192989-b159da00-b200-11e8-8a50-daa2bd45bfc3.png)

* The master node is responsible for aggregating the gradient and Hessian matrix to update the parameters ![equation](http://latex.codecogs.com/gif.latex?W), and sending updated parameters to all the worker nodes.

* ![equation](http://latex.codecogs.com/gif.latex?K) worker nodes are responsible for calculating the partial gradient and Hessian matrix based on the data on this worker, and sending gradient and Hessian information back to the master node. When the new parameters are ready, the worker nodes receive the updated parameters from master node.

## Results
* ### Fast (Linear) Convergence Rete compare to SGD 
![convergence](https://user-images.githubusercontent.com/22249000/44961010-81d46600-aebe-11e8-94d8-df0c6c39f19c.jpg)

* ### Offline Test on 10+ Terabytes training data
![offlinetest](https://user-images.githubusercontent.com/22249000/44961202-f0ff8980-aec1-11e8-8fd7-5f0d8f08e84d.jpg)

* ### 10 days online traffic testing VS Baidu's production (Deep Learning with more machines).
![auccomp](https://user-images.githubusercontent.com/22249000/44961066-851c2180-aebf-11e8-8abc-0ac1a70b333a.png)

## Video (from 52 min)
[![Spark Summit](https://img.youtube.com/vi/mD8EldWuN7k/0.jpg)](https://www.youtube.com/watch?v=mD8EldWuN7k)


## REFERENCE
* Matei Zaharia, Mosharaf Chowdhury, Tathagata Das, Ankur Dave, Justin Ma, Murphy McCauley, Michael J. Franklin, Scott Shenker, Ion Stoica,
"Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing", NSDI 2012. Best Paper Award.

* Jeffrey Dean, Greg S. Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Quoc V. Le, Mark Z. Mao, Marc’Aurelio Ranzato,
Andrew Senior, Paul Tucker, Ke Yang, Andrew Y. Ng, 
"Large Scale Distributed Deep Networks", NIPS 2012.

* Nesterov, Yurii, "Introductory Lectures on Convex Optimization: A Basic Course", Springer, 2014.

* Y. H. Dai, Y. Yuan, "A Nonlinear Conjugate Gradient Method with a Strong Global Convergence Property", SIAM J. Optim, 1999.
