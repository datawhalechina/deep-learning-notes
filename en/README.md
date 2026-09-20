# Deep Learning Notes

## Chapter 1: Introduction to Deep Learning

- 1.1 Neural Networks: A Learnable Function
- 1.2 Loss Function: How Does a Model Know How Wrong It Is?
- 1.3 Forward Propagation, Backpropagation, and Computation Graph
- 1.4 Gradient Descent: From Gradients to Parameter Updates
- 1.5 Why Neural Networks Can Be Trained: Optimization Intuition in High-Dimensional Spaces

## Chapter 2: Getting Started with PyTorch

- 2.1 Automatic Differentiation in PyTorch: From Forward Computation to Backpropagation
- 2.2 Gradient Modes in PyTorch: Controlling How Computation Graphs Are Recorded
- 2.3 Data Loading in PyTorch: Dataset, DataLoader, and Batching
- 2.4 nn.Module in PyTorch: Organizing Models, Parameters, and State
- 2.5 Optimizers in PyTorch: From Manual Updates to Parameter Groups and State Management
- 2.6 Training Loop in PyTorch: Connecting Data, Models, and Optimizers
- 2.7 Checkpoints in PyTorch: Resuming Training After Interruption

## Chapter 3: Multi-Layer Perceptron: From Single Layer to Deep Nonlinear Modeling

- 3.1 From Linear Classifiers to MLPs: Why We Need Hidden Layers
- 3.2 Activation Functions: Adding Nonlinearity to Neural Networks
- 3.3 Softmax and Cross Entropy: From Logits to Classification Loss
- 3.4 Forward and Backward Propagation of Linear Layers
- 3.5 Building a Complete MLP with NumPy
- 3.6 Train MLP on MNIST with NumPy
- 3.7 Backward Propagation Check: Using Numerical Gradients to Verify Handwritten Backward
- 3.8 Reimplementing MLP with PyTorch nn.Module

## Chapter 4: Optimization Algorithms: How Neural Networks Update Parameters

- 4.1 From Gradient Descent to SGD
- 4.2 Momentum and Nesterov Momentum
- 4.3 Adagrad: Starting Point of Adaptive Learning Rates
- 4.4 RMSprop and Adadelta: Fixing Learning Rate Decay
- 4.5 Adam: Combining Momentum and RMSprop
- 4.6 AdamW: Decoupled Weight Decay
- 4.7 Muon: Orthogonalized Updates for Matrix Parameters
- 4.8 Optimizer Map: When to Use Which Optimization Algorithm
- 4.9 Learning Rate Schedulers: Letting the Learning Rate Change During Training

## Chapter 5: Convolutional Neural Networks: From Local Perception to Global Modeling

- 5.1 From MLP to CNN: Why Images Need Convolution
- 5.2 Convolution Computation: Kernel, Padding, Stride, and Channels
- 5.3 Implement Conv2d from Scratch: From Sliding Windows to a PyTorch Module
- 5.4 Pooling and Downsampling: Max Pooling, Average Pooling, and Adaptive Pooling
- 5.5 Building a Simple CNN: From Feature Extraction to Image Classification
- 5.6 LeNet: The Early Template of Convolution, Pooling, and Fully Connected Layers

## Chapter 6: Regularization and Normalization: Making Deep Networks More Stable

- 6.1 Why Deep Networks Need Regularization and Normalization
- 6.2 Dropout: Reducing Overfitting through Random Deactivation
- 6.3 BatchNorm: Stabilizing Training with Batch Statistics
- 6.4 LayerNorm: Normalizing Features Within Each Sample
- 6.5 InstanceNorm: Normalizing Each Channel Within Each Sample
- 6.6 GroupNorm: Normalizing Features Within Channel Groups
- 6.7 RMSNorm: Normalizing Feature Magnitudes Without Mean Centering
- 6.8 A Unified View of Normalization: Which Dimensions Are Normalized?

## Chapter 9: Attention and Transformer: From Dynamic Retrieval to Sequence Modeling

- 9.1 Bahdanau Attention: From Information Compression to Dynamic Retrieval
- 9.2 Cross-Attention: One Sequence Querying Another Sequence
- 9.3 Self-Attention: Internal Information Interaction Within a Sequence
- 9.4 Multi-Head Attention: From Single Perspective to Multiple Perspectives
- 9.5 Positional Encoding: Adding Positional Information to Attention
- 9.6 Transformer Encoder: Stacking Self-Attention Layers
- 9.7 Transformer Decoder: Masked Self-Attention and Cross-Attention
- 9.8 Encoder-Decoder Transformer: Connecting Encoder and Decoder
- 9.9 KV Cache: Why We Don't Recompute the Past During Inference
- 9.10 Three Different Transformer Architectures: Understanding, Generation, and Input-Output Conversion
- 9.11 Hugging Face Transformers API: From Structure to Calls

## Chapter 10: Efficient Attention Implementations: From Memory-Efficient Attention to FlashAttention

- 10.1 Why Attention Is IO-Bound
- 10.2 FlashAttention v1: Eliminating the IO Bottleneck in Attention Mechanisms

## Chapter 11: Implementing GPT from Scratch: From the Transformer Decoder to GPT-2

- 11.1 What Language Models Predict: Next-Token Prediction
- 11.2 MiniGPT: From a Causal GPT Block to a Decoder-only Language Model
- 11.3 Tokenizer: Characters, BPE, and Vocabulary
- 11.4 Embedding, LM Head, and Weight Tying
- 11.5 Training MiniGPT on TinyStories
- 11.6 From Training to Generation: Temperature, Top-k, and Top-p
- 11.7 GPT-2: From MiniGPT to a Pretrained Language Model
