This is my journey of learning and implementing Neural Networks from scratch using NumPy, built to understand how they actually work at fundamental level.

# DAY 1
Today I did a simple implementation of MLP. Here The goal was to understand and do a simple try-out of forward propagation to backpropagation and gradient-based learning.


Disclaimer: The purpose of this program was learning, not performance. It lacks the actual proper use of various function. i went with a simple sigmoid function that assumes all the values to be of range [0,1], and easy to calculate derivative... to simplify things. 

The learning included: 
Random weight initialization, Forward propagation, Sigmoid activation, Sigmoid derivative, Backpropagation, 
Gradient calculation, Gradient descent, Mean Squared Error (MSE), Training over multiple epochs, Prediction after training

It has many limitations starting from gradient descent being applied after every training example to no train/validation or test split and is very basic.(But perfect for my goal).
