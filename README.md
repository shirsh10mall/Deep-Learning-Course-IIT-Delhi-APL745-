# Deep Learning for Mechnanics | Course APL745 | IIT Delhi

## Topics Covered

- **Linear Algebra, Probability, and Numerical Computations:**
  A quick recap of linear algebra concepts, probability theory, and numerical computation methods. These foundational elements are crucial for understanding the mathematical underpinnings of machine learning algorithms.

- **Introduction to Machine Learning:**
  Gain insights into the basics of machine learning, starting with linear regression for predictive modeling. Understand classification techniques to categorize data, and get introduced to the concept of a single-layer perceptron.

- **Deep Feed-Forward Networks:**
  Delve into deep feed-forward neural networks, exploring their architecture and functionality. Learn about learning XOR patterns, gradient-based optimization, hidden units, regularization techniques, and optimization strategies specific to deep learning scenarios.

- **Convolutional Neural Networks (CNNs):**
  Deepen your knowledge with convolutional neural networks (CNNs), a cornerstone of image analysis. Explore the convolution operator for feature extraction, pooling techniques for spatial reduction, various convolutional variants, and their applications in structured output tasks. Understand the neuroscientific basis that underlies the architecture of CNNs.

- **Recurrent Neural Networks (RNNs):**
  Dive into recurrent neural networks (RNNs) that specialize in sequential data analysis. Unpack the unfolding computational graph of RNNs, discover bidirectional RNNs for enhanced context capture, and tackle challenges associated with long-term dependencies. Explore specialized architectures like Long Short-Term Memory (LSTM) networks and other gated RNNs.

- **Physics-Informed Deep Learning:**
  Transition into the realm of physics-informed deep learning, bridging the gap between machine learning and physical sciences.
  - Understand the need for physics-informed approaches in deep learning.
  - Address challenges specific to incorporating physics into neural networks.
  - Learn about strong form and weak form physics-informed methodologies.
  - Explore mixed formulation-based physics-informed deep learning.
  - Apply physics-informed deep learning to time-dependent systems, enabling solutions for complex dynamical problems.

## Assignments

#### Implementation from Scratch (using numpy and scipy)

**1. Univariate and Multivariate Non-Linear Regression for Projectile Motion**
   - Implement univariate and multivariate non-linear regression models using numpy and scipy libraries.
   - Apply these models to solve the projectile motion problem, predicting the trajectory of a projectile under different conditions.

**2. Classification Algorithms with Various Optimization Techniques**
   - Develop binary classification models using techniques like Logistic Regression.
   - Implement One-vs-All multi-class classification and Softmax multi-class classification algorithms.
   - Explore optimization techniques such as Gradient Descent, Stochastic Gradient Descent, and Mini-batch Gradient Descent.

**3. Artificial Neural Network (ANN) for Binary and Multi-class Classification**
   - Build an Artificial Neural Network from scratch using numpy.
   - Apply the ANN to perform binary and multi-class classification tasks using the MNIST dataset.
   - Understand and implement concepts like feedforward propagation, backpropagation, and gradient descent.

**4. Convolutional Neural Network (CNN) for Multi-class Classification**
   - Create a Convolutional Neural Network (CNN) using numpy to tackle image classification problems.
   - Utilize the MNIST dataset to train the CNN model for multi-class classification tasks.
   - Learn about CNN architecture components like convolutional layers, pooling layers, and fully connected layers.

**5. Vanilla Recurrent Neural Network (RNN) for Time Series Data**
   - Implement a basic Recurrent Neural Network (RNN) without specialized libraries.
   - Apply the RNN model to a simple time series dataset, understanding the network's sequential data handling capabilities.

#### Physics Informed Neural Network (PINN)

**1. Solving Differential Equations using Physics Informed Neural Network (PINN)**
   - Implement a Physics Informed Neural Network (PINN) using PyTorch.
   - Apply the PINN to solve differential equations related to static bar problems under specific loading conditions.
   - Solve forward and inverse problems involving differential equations.

**2. Elasticity Partial Differential Equations with PINN**
   - Extend the application of PINN to solving partial differential equations related to the elasticity of a 2D plane.
   - Understand the incorporation of physical principles into neural networks for scientific problem-solving.

**3. DeepONet Architecture for Learning Integration Operator**
   - Implement the DeepONet architecture, a deep neural network designed for learning mathematical operators.
   - Train the neural network to learn the integration operator, showcasing the model's ability to learn fundamental mathematical concepts.

These assignments provide a comprehensive understanding of fundamental machine learning concepts, from implementing regression and classification algorithms from scratch to applying Physics Informed Neural Networks for solving complex differential and partial differential equations. By working through these assignments, you'll gain hands-on experience in building and training various types of neural networks and applying them to real-world problems.

---

**Sources:**
- [Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations](https://www.sciencedirect.com/science/article/pii/S0021999120307991)
- [DeepONet: Learning Operator Networks for Partial Differential Equations](https://arxiv.org/abs/1903.09650)
