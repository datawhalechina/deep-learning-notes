# Deep Learning Notes

## Chapter 1: Introduction to Deep Learning

- 1.1 Neural Networks: A Learnable Function
- 1.2 Loss Function: How Does a Model Know How Wrong It Is?
- 1.3 Forward Propagation, Backpropagation, and Computation Graph

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

## Chapter 11: Vision Transformer: From Image Classification to Visual Sequence Modeling

- 11.1 From CNN to Vision Transformer: Treating Images as Sequences
- 11.2 Patch Embedding: Cutting Images into Tokens
- 11.3 Class Token and Positional Embedding: Letting a Sequence Represent the Whole Image
- 11.4 ViT Encoder: Letting Patch Tokens Exchange Information
- 11.5 ViT Backbone: Pretraining and Fine-Tuning

## Chapter 13: VAE: From Compression and Reconstruction to Probabilistic Generation

- 13.1 AutoEncoder: Starting with Compression and Reconstruction
- 13.2 VAE: Probabilistic Modeling and the Reparameterization Trick
- 13.3 ELBO: Where Does the VAE Objective Function Come From?
- 13.4 VAE Training Phenomena and Latent Space Intuition
- 13.5 VAE: Advantages, Limitations, and Future Developments

## Chapter 14: Diffusion Models: From Denoising to Generation

- 14.1 DDPM: From Denoising to Generation
- 14.2 The Forward Process of DDPM: From Image to Noise
- 14.3 DDPM's Reverse Denoising Process and Training Objective
- 14.4 DDPM Network Structure and Sampling Process
- 14.5 DDPM from a Variational Derivation: Where Does the ELBO Come From?
