# Parallelizing-deep-neural-networks-for-high-frequency-stock-prediction
# Background
Despite the availability of high-frequency stock market data, its use in forecasting stock prices is studied to a lesser extent. Similarly, despite the recent success of neural network on as a forecasting method, its power in forecasting high-frequency dynamics has been relatively overlooked. In addition, most of the studies in the literature have been focused on stock market indices instead of individual stocks. A possible explanation is the intractable computational intensity of training neural networks on the massive volume of high-frequency data of individual stocks. This motivates our study on applying parallelism to the training task and evaluate its performance to demonstrate weak and strong scaling.

Parallel neural network has also been a topic of great interest. There are generally two paradigms of parallelisation: data parallelism and model parallelism. Data Parallel is straightforward. Its correctness is mathematically supported, and is very commonly implemented with MPI [4]. Model parallel is more complicated. For large models that include millions to billions of parameters, the whole neural network is partitioned across different machines, and usually implemented with openMP and GPU. For our intended purpose, we can fit the whole network into one machine so parallelism is applied to the largest computational bottleneck - BLAS operations in backpropagation algorithm.

# Data
We formulate the task as a prediction problem, using lagged previous prices of individual stocks to predict future prices at the minute level. The high-frequency consolidated trade data for the US equity market comes from NYSE Trade and Quote (TAQ) database, available by the WRDS research center.

Specifically, the inputs are price and volume information at or before minute t for all stocks except stock j. Technical indicators of price series includes:

Exponential Moving Averages (EMA) and Moving Averages (MA)
Past k-period log returns
PSY: fraction of upward movement in the past k-period
Price and returns volatility over k periods
Momentum: Change in price in k periods
Disparity: last available price over MA
The output is the predicted return at minute t+1 for a stock. We normalize all the input and output variables using z-score and unit norm per feature.

# Methods
Neural Network Architecture
For the prediction method, multi-layer Artificial Neural Networks (ANN) using back-propagation algorithm has shown promising results in stock index prices compared with traditional methods [1]. Note that the traditional gradient descent algorithm of back-propagation is sequential by nature. We will therefore apply a technique that combines MPI with OpenMP/CUDA for BLAS to parallelize the training process: asynchronized multiple sub-neural networks[3] with nested parallel batch Stochastic Gradient Descent[2].

The goal of our project is to implement a two-level parallelization model by combining MPI and OpenMP/CUDA. Unfortunately, developing executable code using OpenMP (via Cython) resulted in an onerous and difficult task, therefore, we opted for existing optimization algorithms for the update of gradients, then analyzing the nature of our algorithm by comparing time until convergence as well as the average runtime for each training iteration at fixed batch sizes. Nonetheless, we describe our desired design and the design we used for our project below.

Neural Network Architecture (hyper-parameters)
We implement a fully connected network with:

L = 4 layers
number of neurons = 42,24,12,1; fewer neurons in deeper layers (pyramidal architecture)
Gradient-based and non-gradient-based optimizers (AdaGrad, Hessian Free, and Particle Swarm Optimization)
ReLu/MSE activation, linear activation for output node
L2 and maxnorm regularization, early stopping, dropouts
Parallelism Architecture
We execute data and model parallelism at two levels. Firstly, each machine (e.g. an Odyssey node) will store a Data Shard (a subset of data) and train a model replica independently and asynchronously (see Figure 1.) Each replica will fetch weights (𝑤) from the parameter server (the master node), compute ∆𝑤, and push ∆𝑤 to the server or master node. The parameter server updates the parameter set whenever it receives ∆𝑤 from a model replica. This architecture is reasonable because the updating and validation process involves much less computation than back-propagation in the model replicas. We analyzed three different optimization algorithms for the update of the weights. The fetching and pushing weights and gradient weights to the master node was implemented with MPI (mpi4py package).

![image](https://user-images.githubusercontent.com/63738424/164894584-1f3b4786-4dd5-41ad-8ec2-b03239523f61.png)


Figure 1: Parallelised Neural Network Architecture . Model replicas asynchronously fetch parameters 𝑤 and push ∆𝑤 to the parameter server.

Secondly, each model replica aimed to compute ∆𝑤 by averaging the mini-batch gradients from 64 or 32 (depend on number of cores in a node) parallel threads (see Figure 2). We attempted to implement this level of parallelism with OpenMP (Cython parallel module). However, we were unsuccessful with this implementation, so we used OpenMP/CUDA for BLAS in each model replica (to parallel matrix computations) and tested at different cores.



![image](https://user-images.githubusercontent.com/63738424/164894604-6595e3a7-6c70-478d-8318-47a73b4af366.png)

Figure 2: Desired parallelization in each model replica.

![image](https://user-images.githubusercontent.com/63738424/164894632-1e1eca47-7614-411f-9f8e-7e7263331855.png)


Figure 3: Real architecture of our algorithm. Note that node 0 is the master node, where the optimization takes place, and node 1 through 7 (number of total nodes can and will vary) is a model replica, where the calculation of the gradient of weight occurs.

# Optimization methods
Adaptive Gradient Algorithm (AdaGrad): modified SGD with parameter learning rate. Informally, this increases the learning rate for more sparse parameters and decreases the learning rate for less sparse ones. This strategy improves convergence performance where data is sparse. This optimization method is run with MPI architecture (see Figure 3). Specifically, there are two message passing schemes suitable for different hardware settings:

Instant updating: each model replica communicates with the parameter server for every batch iteration. This scheme is suitable when the communication cost between nodes is relatively small, e.g. the seas_iacs partition on Odyssey. It is very close to the sequential version, and converges most quickly (least number of iterations) since it makes use of information from all data shards.
Cumulative updating: each model replica communicates with the parameter server after a few (e.g. 20) batch iterations. This scheme is suitable when the communication cost is large since fewer message passings are needed here. However, it is possible for a model replica to get over-fitted since it only has a subset of data.
Hessian-Free (Truncated Newton Method): an approximation of the Hessian is calculated, which saves time and computational resources, when updating using the well known Newton method. However, this method updates the model parameters sequentially and does not naturally fit into our MPI parallel architecture. Therefore, we implemented a standalone version with one level of parallelisation (using GPU as a feature of the hessionfree pacakge).

Particle Swarm Optimization (PSO): computational method that solves a problem by having a population of candidate solutions, or particles, and moving these around in the search-space according to simple mathematical formulae over the particle's position and velocity. Each particle's movement is influenced by its local best known position, but is also guided toward the best known positions in the search-space, which are updated as better positions are found by other particles. This is expected to move the swarm toward the best solutions.

Hybrid system (global + local search): Initialized swarms with AdaGrad, then update swarms with PSO methods

# Done By
S.HARIHARA SUDHAN--19BCE0742

AMAN SINHA--19BCE0706

PRASHANT MAIKHURI--19BCE2476
