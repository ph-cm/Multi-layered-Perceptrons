# Multi-Layered Perceptrons
## Project Overview

In this project, we will progressively build a simple neural framework from scratch. This framework will be capable of handling both **multi-class classification** and **regression tasks** using multi-layered perceptrons (MLPs), a type of feedforward neural network.

### Key Objectives
- **Understand** the fundamentals of multi-layered perceptrons and their role in machine learning.
- **Implement** a neural framework capable of solving various supervised learning tasks.
- **Learn** the underlying mathematics and coding techniques for building and training MLPs.

### Prerequisites
This project assumes a basic understanding of Python programming, linear algebra, and calculus. Familiarity with neural networks and machine learning is helpful but not required.

### Project Structure
- **Data Preparation:** Handling and preprocessing input data for neural networks.
- **Model Construction:** Building layers, activation functions, and loss functions for MLPs.
- **Training Loop:** Implementing forward and backward propagation, calculating gradients, and updating weights.
- **Evaluation Metrics:** Measuring performance on classification and regression tasks.

### Generating, Visualizing, and Analyzing a Synthetic Dataset
In this guide, we'll explain how to create a synthetic dataset and visualize it using Python. The goal is to demonstrate how classification models can be trained and tested using a simple dataset. This example covers key concepts like data generation, splitting into training and testing sets, and visualizing data using scatter plots.
- Generating a dataset using `make_classification` from `sklearn`.
- Splitting the dataset into training and testing sets.
- Visualizing the dataset using scatter plots.
- Inspecting sample data for verification.

### Machine Learning Problem
Machine learning revolves around creating models that can learn patterns from data to make predictions or decisions. The goal is to optimize the model to generalize well on unseen data by minimizing a specific loss function.

## 📊 Dataset
Suppose we have an input dataset \((X, Y)\), where:
- \(X\) is a set of features (inputs).
- \(Y\) is the corresponding set of labels (outputs).

The type of labels depends on the problem being solved:
- For **regression** tasks: \( y_i \in \mathbb{R} \) (i.e., labels are real-valued).
- For **classification** tasks: Labels are represented as discrete classes \( y_i \in \{0, 1, ..., n\} \).

## 🧠 Model Representation
In machine learning, a model can be represented as a function:

\[
f_{\theta}(x)
\]

Where:
- \( \theta \) represents the parameters that the model learns during the training process.
- The objective is to adjust \( \theta \) so that the model fits the training data as accurately as possible.

## 🎯 Optimization Goal
To optimize our model, we need to minimize a **loss function** \( \mathcal{L} \). This is done by finding the optimal parameters \( \theta \) that result in the best fit:

\[
\theta = \arg \min_{\theta} \mathcal{L}(f_{\theta}(X), Y)
\]

- The loss function \( \mathcal{L} \) quantifies the difference between the model's predictions and the actual labels.
- The optimization goal is to find the parameters \( \theta \) that minimize this difference.

## ⚙️ Loss Function
The choice of the loss function depends on the type of problem being addressed:

- For **regression tasks**:
  - **Mean Squared Error (MSE)**
  - **Mean Absolute Error (MAE)**
- For **classification tasks**:
  - **Cross-Entropy Loss**
  - **Hinge Loss**

The loss function guides the model on how to adjust its parameters to improve performance.

### 🔍 Loss Functions for Regression
For regression tasks, two commonly used loss functions are:
1. **Absolute Error** \( \mathcal{L}_{\text{abs}}(\theta) \):
   \[
   \mathcal{L}_{\text{abs}}(\theta) = \sum_{i=1}^{n} |y_i - f_{\theta}(x_i)|
   \]
   - Measures the absolute difference between the predicted and actual values.
   - Less sensitive to outliers compared to squared error.

2. **Mean Squared Error** \( \mathcal{L}_{\text{sq}}(\theta) \):
   \[
   \mathcal{L}_{\text{sq}}(\theta) = \sum_{i=1}^{n} (y_i - f_{\theta}(x_i))^2
   \]
   - Measures the squared difference between the predicted and actual values.
   - More sensitive to outliers, as errors are squared.

### 🔍 Loss Functions for Classification
Let's consider a **binary classification** problem where the labels are either `0` or `1`. The output of the model \( f_{\theta}(x_i) \in [0, 1] \) represents the probability of choosing class `1`.

#### 1. **0-1 Loss**
The **0-1 loss** function simply counts the number of correct classifications:

\[
\mathcal{L}_{0-1} = \sum_{i=1}^{n} l_i, \quad l_i = 
\begin{cases} 
0 & \text{if } (f(x_i) < 0.5 \land y_i = 0) \lor (f(x_i) \geq 0.5 \land y_i = 1) \\
1 & \text{otherwise}
\end{cases}
\]

- This loss is equivalent to calculating the **accuracy** of the model.
- It simply checks whether the predicted output matches the true label.

However, accuracy alone does not reflect how far off the predictions are from the correct class. A prediction that is very close to the decision boundary (e.g., 0.49 for class `0` when the threshold is 0.5) is treated the same as a prediction far away from it. Therefore, **logistic loss** is often preferred.

#### 2. **Logistic Loss**
The **logistic loss** (also known as **cross-entropy loss**) is commonly used in binary classification. It provides a continuous measure of how well the model's predictions match the true labels:

\[
\mathcal{L}_{\text{log}} = \sum_{i=1}^{n} \left[-y_i \log(f_{\theta}(x_i)) - (1 - y_i) \log(1 - f_{\theta}(x_i))\right]
\]

- Penalizes incorrect classifications more severely than the 0-1 loss.
- Ensures that predictions closer to the correct class are rewarded, while those further away are penalized.
- The logistic loss is widely used in classification tasks due to its effectiveness in optimizing probabilistic models.

To understand how logistic loss works, consider two cases:
- If we expect the output to be `1` (\( y = 1 \)), then the loss is \( -\log(f_{\theta}(x_i)) \). The loss is `0` if the model predicts `1` with a probability of `1`, and it grows larger as the probability of predicting `1` decreases.
- If we expect the output to be `0` (\( y = 0 \)), then the loss is \( -\log(1 - f_{\theta}(x_i)) \). Here, \( 1 - f_{\theta}(x_i) \) is the probability of predicting class `0`. The closer this probability is to `1`, the lower the loss.

### 🖥️ Neural Network Architecture
We have generated a dataset for a binary classification problem. However, let's generalize it to **multi-class classification** from the beginning. This way, we can easily switch between binary and multi-class classification.

In this case, our one-layer perceptron (a basic neural network with a single hidden layer) will have the following architecture:

1. **Input Layer**: Receives the features \( X \).
2. **Hidden Layer**: Applies a non-linear activation function (e.g., ReLU).
3. **Output Layer**:
   - For binary classification: A single neuron with a **sigmoid** activation function.
   - For multi-class classification: Multiple neurons (one for each class) with a **softmax** activation function.

The output layer's configuration will adjust automatically based on the classification task, making this architecture flexible.

## 🌐 Softmax: Turning Outputs into Probabilities

When working with neural networks, especially in multi-class classification, the raw outputs (logits) are not necessarily probabilities. They can take on any value, positive or negative. To interpret these outputs as probabilities, we need to normalize them such that they sum to 1 across all classes. This is where the **Softmax function** comes into play.

### 🔢 Softmax Formula
The Softmax function converts the output \( z_c \) for each class \( c \) into a probability:

\[
\sigma(z_c) = \frac{e^{z_c}}{\sum_{j} e^{z_j}}, \quad \text{for } c \in \{1, \ldots, |C|\}
\]

- \( \sigma(z_c) \): Probability of class \( c \).
- \( z_c \): Raw output (logit) for class \( c \).
- \( \sum_{j} e^{z_j} \): Sum of exponentials of all logits, ensuring that the resulting values are normalized to form a valid probability distribution.

### 🛠️ Why Use Softmax?
- The Softmax function ensures that the model's outputs are interpreted as probabilities, making them suitable for multi-class classification tasks.
- The class with the highest Softmax value is typically chosen as the predicted class, as it represents the most likely outcome.

## 📉 Cross-Entropy Loss

In classification tasks, one of the most commonly used loss functions is the **cross-entropy loss**. This loss function is a generalization of the **logistic loss** and is particularly effective in comparing two probability distributions.

### 🔍 What is Cross-Entropy Loss?
The cross-entropy loss measures the similarity between the predicted probability distribution output by the model and the true distribution (often represented as a **one-hot** vector). In essence, it penalizes the model when the predicted probability for the correct class is low.

### 🔢 Formula
For a given class \( c \) with probability output \( p_c \) from the model, the cross-entropy loss can be calculated as:

\[
\text{Loss} = -\log(p_c)
\]

Where:
- \( p_c \) is the predicted probability of the correct class \( c \).
- If the model predicts \( p_c = 1 \) for the correct class, the loss is zero.
- If the model predicts \( p_c \) close to zero for the correct class, the loss approaches infinity.

### 🛠️ How Cross-Entropy Works
In our case:
- The first distribution is the **probabilistic output** of the neural network (using Softmax).
- The second distribution is the **one-hot** distribution, where the correct class has a probability of 1, and all others have a probability of 0.

The cross-entropy loss is minimized when the predicted probability for the correct class is close to 1. The further away the predicted probability is from 1 for the correct class, the higher the loss will be. This encourages the model to become more confident in its predictions.

## 🛠️ Loss Minimization Problem and Network Training

Now that we have defined our neural network as \( f_{\theta} \) and established a loss function \( \mathcal{L}(Y, f_{\theta}(X)) \), we can think of \( \mathcal{L} \) as a function of the model parameters \( \theta \). Our goal in training the network is to minimize this loss function:

\[
\theta = \arg \min_{\theta} \mathcal{L}(Y, f_{\theta}(X))
\]

### 📉 Gradient Descent
To minimize the loss function, we use **Gradient Descent**, a well-known optimization algorithm. The core idea is to adjust the parameters \( \theta \) iteratively in the opposite direction of the gradient (partial derivatives) of the loss function to reduce the error.

Gradient descent works as follows:
1. **Initialize** parameters \( W \) and \( b \) randomly.
2. **Repeat** the following steps until convergence:

\[
W^{(t+1)} = W^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial W} \tag{1}
\]
\[
b^{(t+1)} = b^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial b} \tag{2}
\]

- \( \eta \) is the learning rate.
- \( \frac{\partial \mathcal{L}}{\partial W} \) and \( \frac{\partial \mathcal{L}}{\partial b} \) are the gradients with respect to \( W \) and \( b \).

### 🚀 Stochastic Gradient Descent (SGD)
In practice, instead of using the entire dataset to compute gradients, we use smaller subsets called **minibatches**. This technique is known as **Stochastic Gradient Descent (SGD)**. It helps in speeding up the training process and reducing memory usage.

---

## 🔄 Backpropagation

To efficiently compute the gradients needed for training, we use a technique called **backpropagation**. This algorithm allows us to compute the derivatives of the loss function with respect to each weight \( W \) and bias \( b \) in the network.

### 🔁 Understanding Backpropagation
To compute \( \frac{\partial \mathcal{L}}{\partial W} \) and \( \frac{\partial \mathcal{L}}{\partial b} \), we use the **chain rule** for derivatives. Here’s how it works:

1. Suppose we obtain a loss \( \Delta \mathcal{L} \) from the output.
2. To minimize the loss, we adjust the Softmax output \( p \) by \( \Delta p = \left(\frac{\partial \mathcal{L}}{\partial p}\right) \Delta \mathcal{L} \).
3. We adjust the hidden layer output \( z \) by \( \Delta z = \left(\frac{\partial p}{\partial z}\right) \Delta p \).
4. Finally, adjust the weights and biases:
   - \( \Delta W = \left(\frac{\partial z}{\partial W}\right) \Delta z \)
   - \( \Delta b = \left(\frac{\partial z}{\partial b}\right) \Delta z \)

### 🛠️ Two-Step Process in Training
Each training iteration consists of:
1. **Forward Pass**: Calculate the loss for a given input batch.
2. **Backward Pass**: Minimize the loss by distributing the error back through the network using the gradients.

---

## ⚙️ Implementation of Backpropagation

To implement backpropagation, we add a `backward` function to each layer in our network. This function calculates the derivatives needed to adjust the weights and biases.

### 🔢 Derivatives Calculation
For a linear layer defined as \( z = x \times W + b \):

\[
\frac{\partial z}{\partial W} = x \tag{5}
\]
\[
\frac{\partial z}{\partial b} = 1 \tag{6}
\]

If we need to adjust for the error \( \Delta z \) at the output layer, we update the weights and biases as follows:

\[
\Delta x = \Delta z \times W \tag{7}
\]
\[
\Delta W = \left(\frac{\partial z}{\partial W}\right) \Delta z = \Delta z \times x \tag{8}
\]
\[
\Delta b = \left(\frac{\partial z}{\partial b}\right) \Delta z = \Delta z \tag{9}
\]

### 🔄 Minibatch Updates
The calculations are not done for each training sample individually but rather for a whole **minibatch**. The required updates \( \Delta W \) and \( \Delta b \) are computed across the minibatch:

- \( x \in \mathbb{R}^{\text{minibatch} \times \text{n\_class}} \)

## 🚀 Training the Model

Once we have defined the network and implemented backpropagation, we are ready to train it. The training process involves iterating over the dataset multiple times and optimizing the model using **minibatches**.

### 📆 Training Loop
The **training loop** goes through the entire dataset repeatedly to optimize the network parameters. One complete pass through the dataset is called an **epoch**. During each epoch:
1. The data is split into smaller batches (**minibatches**).
2. For each minibatch:
   - Perform a **forward pass** to compute predictions.
   - Calculate the **loss** using the predictions.
   - Perform a **backward pass** to update the weights using the gradients.

## 🧩 Network Class

In many cases, a neural network is just a composition of layers. By building a **Network Class**, we can stack layers together and seamlessly handle the forward and backward passes through the network without explicitly coding these operations for each layer.

### 🔧 Why Use a Network Class?
- It simplifies the process of creating and training neural networks.
- Allows easy stacking of different layers like dense (fully connected), activation, and output layers.
- Automates the forward and backward passes, making the code modular and reusable.

### 🛠️ Implementation
We will store a list of layers inside the `Net` class and use the `add()` function to add new layers. The `Net` class manages both forward propagation and backpropagation for all the layers it contains.

## 📊 Plotting the Training Process

To better understand how the neural network is being trained, it can be helpful to visualize its progress. For this, we will define a function called `train_and_plot` that allows us to observe how the model's decision boundaries change as it learns.

### 🖼️ Visualization Approach
- We will use a **level map** to represent different output values of the network using colors.
- This approach provides a visual representation of the regions where the network predicts different classes.
- By plotting the decision boundaries at various stages of training, we can gain insights into how the model is improving.

## 🏗️ Multi-Layered Models

In our previous examples, we constructed neural networks using a single **Linear** layer for classification. However, what happens if we decide to add multiple layers?

The good news is that our existing code will still work! However, there's an important concept to understand: simply stacking multiple linear layers will not increase the network's expressive power. This is because the composition of linear functions is still a linear function. To build more powerful models, we need to introduce **non-linear activation functions**, such as the `tanh` function.

### 🧠 Why Add Multiple Layers?
- **Non-linearity**: By adding non-linear activation functions between linear layers, the network gains the ability to model complex, non-linear relationships in the data.
- **Expressiveness**: A network with multiple layers and non-linearities can classify data that is not linearly separable. For example:
  - A two-layer network can classify any convex set of data points.
  - A three-layer network can classify virtually any set of data points.

## ❓ Why Not Always Use a Multi-Layered Model?

We have seen that a **multi-layered model** is generally more powerful and expressive than a simple one-layer network. However, you may be wondering: **why don't we always use a many-layered model?** The answer to this question lies in a concept known as **overfitting**.

### 📉 Understanding Overfitting
While a more complex model can fit the training data better, it also requires more data to properly generalize to new, unseen data. The more powerful the model, the higher the risk of it becoming **too specialized** to the training dataset, leading to overfitting.

- **Overfitting** occurs when a model performs well on the training data but fails to generalize to new, unseen data.
- **Underfitting** is the opposite, where the model is too simple to capture the patterns in the training data.

---

### 🔍 A Closer Look

#### 1. A Linear Model
- **Characteristics**:
  - Likely to have a **high training loss** due to its limited expressiveness — this is known as **underfitting**.
  - Validation loss and training loss are generally similar, indicating that the model generalizes well to test data.
- **Conclusion**:
  - The model lacks the power to capture complex patterns but is less prone to overfitting.
  - Suitable for simpler tasks where the data is mostly linearly separable.

#### 2. Complex Multi-Layered Model
- **Characteristics**:
  - **Low training loss**: The model can approximate the training data well because it has enough expressive power.
  - **High validation loss**: The validation loss can be much higher than the training loss, especially if the model is too complex. This happens because the model **memorizes the training points** and loses the ability to generalize.
- **Conclusion**:
  - While powerful, multi-layered models require more data and regularization techniques to avoid overfitting.
  - Useful for complex tasks where the data has non-linear patterns.

---

### 🗝️ Key Takeaways
- **Model Complexity vs. Generalization**: The more powerful the model, the better it can fit the training data. However, with great power comes the risk of overfitting.
- **Balancing Act**: It's crucial to find the right balance between model complexity and the amount of data available.
- **Practical Implications**:
  - Use simpler models for smaller datasets to avoid overfitting.
  - Reserve complex, multi-layered architectures for problems where you have a large amount of training data and complex patterns to capture.

### 🛠️ Strategies to Prevent Overfitting
To mitigate overfitting, consider using techniques such as:
- **Regularization** (L1, L2 penalties)
- **Dropout layers** in neural networks
- **Early stopping** during training
- **Cross-validation** to evaluate model performance on unseen data

