# Changelog

## August 2026 Release

This release expands the bilingual deep-learning curriculum with new and refined material on loss functions, neural-network trainability, convolutional networks, VAEs, diffusion models, attention, MLPs, memory engineering, and LLM training engineering. It also adds a complete Stanford CS336 Assignment 1 implementation, substantially improves dnnlpy, and modernizes the project’s build, rendering, and release infrastructure.

### New Notebooks

#### Chapter 1: Introduction to Deep Learning

- 1.2 Loss Function: How Does a Model Know How Wrong It Is?
- 1.3 Forward Propagation, Backpropagation, and Computation Graph [rewrite]
- 1.5 Why Neural Networks Can Be Trained: Optimization Intuition in High-Dimensional Spaces

#### Chapter 5: Convolutional Neural Networks

- 5.1 From MLP to CNN: Why Images Need Convolution
- 5.2 Convolution: Kernels, Padding, Stride, and Channels
- 5.3 Implementing Conv2d from Scratch: From Sliding Windows to PyTorch Modules
- 5.4 Pooling and Downsampling: Max, Average, and Adaptive Pooling
- 5.5 Building a Simple CNN: From Feature Extraction to Image Classification
- 5.6 LeNet: An Early Template for Convolution, Pooling, and Fully Connected Layers

#### Chapter 14: Diffusion Models: From Denoising to Generation

- 14.1 DDPM: From Denoising to Generation [rewrite]

#### Chapter 19: LLM Training Engineering

- 19.1 Where Does LLM Training Memory Go? Model States, Activations, and Runtime Overhead

### Stanford CS336 Assignment 1

- Added a complete Transformer language-model implementation for Assignment 1.
- Added linear layers, embeddings, RMSNorm, SwiGLU, Rotary Position Embeddings, multi-head self-attention, Transformer blocks, and a Transformer language model.
- Added data loading, cross-entropy, gradient clipping, AdamW, checkpoint serialization, and learning-rate scheduling utilities.
- Added a compatible BPE tokenizer with byte-level processing, special-token support, encoding, decoding, and training.
- Added an optimized incremental BPE training implementation.
- Expanded the assignment write-up with profiling results, implementation analysis, and Mermaid diagrams.
- Added comprehensive reference fixtures, snapshots, and tests derived from the assignment interface.

### `dnnlpy` Package Updates

- Added one-, two-, and three-dimensional max pooling and average pooling modules.
- Added adaptive max pooling and adaptive average pooling APIs.
- Added matching functional pooling implementations and expanded public exports.
- Added device-memory inspection and cleanup utilities.
- Added reusable byte-conversion utilities.
- Added CS336 Assignment 1 model and utility modules.
- Improved traditional tokenizer training with persistent model state.
- Improved tokenizer compatibility across supported Python and dependency versions.
- Aligned BPE tokenization behavior more closely with Hugging Face tokenizers.
- Refined tokenizer vocabulary, merge, batch-encoding, and special-token handling.
- Improved `Trainer` metric collection, evaluation history, logging, and state tracking.
- Modernized model, optimizer, activation, folding, convolution, and gradient-clipping APIs.
- Renamed the linear-layer parameter attributes to `weight` and `bias` and aligned the MLP tests with the public API.
- Expanded tests for pooling, tokenizers, models, optimizers, training utilities, and public exports.
- Adjusted the TinyStories tokenizer performance test for more reliable execution across environments.
- Updated project metadata and dependencies for PyTorch 2.13 and Transformers 5.15.
- Marked `dnnlpy` as stable and finalized its version as 2026.08.24.

### Book and Documentation Updates

- Renumbered the Attention and Transformer material from Chapter 8 to Chapter 9.
- Removed the incomplete GAN and vision-language-model chapters from the current book structure.
- Revised foundational PyTorch, MLP, optimization, efficient-attention, Vision Transformer, VAE, diffusion, regularization, and GPT-2 material.
- Refreshed bilingual diagrams and standardized Mermaid source and SVG naming.
- Updated installation instructions, repository links, contribution guidance, citation metadata, and project roadmaps.
- Added an AI-assisted-writing disclosure, book-author and website metadata, and cover-design attribution.
- Improved the CS336 notebook references and generated notebook artifacts.
- Added a new book cover and refreshed Typst styling for code, quotations, and bilingual PDF output.
- Added dedicated Typst reference-section handling and corrected part-title sizing and table-header alignment in the PDF output.
- Added the missing English chapters and aligned them with the Chinese version.

### Build and Development Updates

- Added a Dockerfile and development-container configuration.
- Added pre-commit configuration and refreshed Ruff and Python tooling.
- Added Markdown linting configuration and expanded Ruff formatting coverage for Markdown, Quarto, and notebook files.
- Updated Quarto profiles for HTML, notebook, and bilingual Typst rendering.
- Bumped the minimum Quarto version to 1.10.
- Added Posit Connect Cloud publishing configuration.
- Simplified cache, checkpoint, attachment, image-attribute, table-of-contents, and PDF-renaming utilities.
- Removed committed Quarto freeze artifacts and consolidated cache cleanup.
- Updated dataset-download and Mermaid-cleanup tooling.
- Finalized the project and `dnnlpy` versions as 2026.08.24 and marked the release metadata as stable.

### CI Updates

- Consolidated website, notebook, PDF, and pull-request rendering workflows.
- Added manually dispatched website and bilingual PDF rendering workflows.
- Added Docker image release automation.
- Improved platform-aware execution, caching, artifact packaging, and release handling.
- Standardized workflow concurrency behavior and restored published-release triggers for package and Docker publishing.
- Restricted PyPI, TestPyPI, and notebook-sync side effects to the canonical `jshn9515/deep-learning-notes` repository.
- Made matrix virtual-environment creation independent of the root project’s Python requirement.
- Updated GitHub Actions dependencies, including newer Python, uv, checkout, and cache actions.
- Simplified dnnlpy testing and packaging automation.

### Merged Pull Requests

- DEP: Update dependency transformers to >=5.14.1,<5.15.0 by @renovate[bot] in [#2](https://github.com/jshn9515/deep-learning-notes/pull/2)
- DEP: Update dependency transformers to >=5.15.0,<5.16.0 by @renovate[bot] in [#6](https://github.com/jshn9515/deep-learning-notes/pull/6)
- DEP: Update astral-sh/setup-uv action to v10 by @renovate[bot] in [#7](https://github.com/jshn9515/deep-learning-notes/pull/7)
- DEP: Update dependency tiktoken to >=0.14.0,<0.15.0 by @renovate[bot] in [#9](https://github.com/jshn9515/deep-learning-notes/pull/9)
- [zh] FIX: Correct InstanceNorm example input usage by @wqpwqp1222 in [#11](https://github.com/jshn9515/deep-learning-notes/pull/11)
- MNT: Simplify GroupNorm statistics example by @wqpwqp1222 in [#12](https://github.com/jshn9515/deep-learning-notes/pull/12)
- DEP: Update dependency diffusers to >=0.40.0,<0.41.0 by @renovate[bot] in [#13](https://github.com/jshn9515/deep-learning-notes/pull/13)
- DOC: Improve the wording and clarity of the tutorial content by @tiansiyuan in [#65](https://github.com/datawhalechina/deep-learning-notes/pull/65)
- [zh] FIX: Correct conv2d output size example variable name by @wqpwqp1222 in [#69](https://github.com/datawhalechina/deep-learning-notes/pull/69)
- [zh] DOC: Clarify Conv2d padding and fix math formatting by @wqpwqp1222 in [#70](https://github.com/datawhalechina/deep-learning-notes/pull/70)

### New Contributors

- @renovate[bot] made their first contribution in [#2](https://github.com/jshn9515/deep-learning-notes/pull/2)
- @wqpwqp1222 made their first contribution in [#11](https://github.com/jshn9515/deep-learning-notes/pull/11)
- @tiansiyuan made their first contribution in [#65](https://github.com/datawhalechina/deep-learning-notes/pull/65)

> [!NOTE]
> This project continues to be maintained in both **Chinese** and **English** through a Quarto-based structure, as an open and continuously growing collection of deep learning study notes.

**Full Changelog**: [https://github.com/jshn9515/deep-learning-notes/compare/v2026.07.21...v2026.08.24](https://github.com/jshn9515/deep-learning-notes/compare/v2026.07.21...v2026.08.24)

## July 2026 Release

This release introduces new chapters on regularization and normalization, GPT-2 implementation, and Stanford CS336 language-modeling fundamentals. It also renames the accompanying library from `dnnl` to `dnnlpy` and substantially expands its neural-network, tokenizer, training, and model APIs.

### New Notebooks

#### Chapter 7: Regularization and Normalization

- 7.1 Why Deep Networks Need Regularization and Normalization
- 7.2 Dropout: Reducing Overfitting Through Random Deactivation
- 7.3 BatchNorm: Stabilizing Training with Batch Statistics
- 7.4 LayerNorm: Normalizing Features Within Each Sample
- 7.5 InstanceNorm: Normalizing Each Channel Within a Sample
- 7.6 GroupNorm: Normalizing Features Within Channel Groups
- 7.7 RMSNorm: Feature-Scale Normalization Without Mean Centering
- 7.8 A Unified View of Normalization Methods

#### Chapter 18: Implementing GPT-2 from Scratch

- 18.1 Next-Token Prediction: What Language Models Learn to Predict
- 18.2 MiniGPT: From Causal GPT Blocks to a Language Model
- 18.3 Tokenizers: Character Tokenization, BPE, and Vocabularies
- 18.4 Embeddings, Language-Modeling Heads, and Weight Tying
- 18.5 Training MiniGPT on TinyStories
- 18.6 From Training to Generation: Temperature, Top-k, and Top-p Sampling
- 18.7 GPT-2: From MiniGPT to a Pretrained Language Model

#### Stanford CS336: Language Modeling from Scratch

- Added a dedicated CS336 course section.
- Added the Assignment 1 write-up covering the implementation of a small language model from scratch.
- Added BPE tokenizer training scripts for TinyStories and OpenWebText.
- Included the original assignment handout and supporting configuration.

### `dnnlpy` Package Updates

- Renamed the Python package from `dnnl` to `dnnlpy` for PyPI distribution.
- Added a complete tokenizer subsystem with character, word, and BPE tokenizers.
- Added byte-level and whitespace pre-tokenizers, normalizers, decoders, post-processors, batch encoding, and BPE training utilities.
- Improved BPE training behavior, including preservation of special-token IDs and more robust vocabulary, merge, and special-token handling.
- Added the MiniGPT model with causal attention, positional embeddings, weight tying, residual projection scaling, and token-sampling utilities.
- Added generation utilities supporting temperature, top-k, and top-p sampling.
- Expanded neural-network modules with convolution, bilinear, embedding, flattening, folding, dropout, normalization, activation, and loss layers.
- Added functional implementations for convolution, normalization, regularization, representation, folding, activation, attention, affine transformations, and loss operations.
- Added GLU and SwiGLU activations and improved numerical stability for sigmoid, softplus, and related functions.
- Expanded normalization support with BatchNorm, InstanceNorm, LayerNorm, GroupNorm, RMSNorm, and Local Response Normalization.
- Expanded Transformer and attention components, including positional-embedding utilities, basic attention, fast projection paths, and improved module representations.
- Added gradient-clipping utilities.
- Added `LinearLR`, `ConstantLR`, and `CosineAnnealingLR` learning-rate schedulers.
- Added a reusable `Trainer` with checkpointing, resumption, gradient clipping, logging, maximum-step limits, and maximum-time limits.
- Updated ViT, DDPM, VAE, Seq2Seq, MLP, and Transformer models to use reusable `dnnlpy` components.
- Expanded test coverage across models, tokenizers, neural-network modules, optimizers, schedulers, gradient clipping, and training utilities.
- Added opt-in slow-test support for dataset-backed tokenizer tests.
- Updated package metadata and dependencies for Python 3.12, 3.13, and 3.14.

### Documentation Updates

- Revised existing English and Chinese chapters covering PyTorch fundamentals, MLPs, optimization, attention, efficient attention, Vision Transformers, generative models, and vision-language models.
- Added and refreshed Mermaid architecture diagrams across multiple chapters.
- Added project architecture documentation describing the website, notebook, and PDF build pipelines.
- Improved notebook conversion, attachment cleanup, image handling, and PDF formatting.
- Added dedicated Quarto profiles for HTML, Jupyter Notebook, and Typst PDF output.

### CI Updates

- Replaced the original `dnnl` workflow with a dedicated `dnnlpy` test and build pipeline.
- Added automated PyPI and TestPyPI release workflows.
- Added testing across Python 3.12, 3.13, and 3.14 before package publication.
- Added Ruff formatting checks to package release workflows.
- Added bilingual Typst PDF rendering with release artifact uploads.
- Added artifact attestations for packaged notebooks and rendered PDFs.
- Updated notebook packaging to use the Quarto Jupyter profile.
- Added automatic synchronization of packaged English and Chinese notebooks with the `dnnl-notebooks` repository.
- Improved release artifact handling for notebook archives and PDFs.
- Updated GitHub Actions dependencies and reorganized publishing and pull-request rendering workflows.

### Merged Pull Requests

- DEP: Update dependency matplotlib to >=3.11.0,<3.12.0 by @renovate[bot] in [#27](https://github.com/datawhalechina/deep-learning-notes/pull/27)
- DEP: Update dependency accelerate to >=1.14.0,<1.15.0 by @renovate[bot] in [#26](https://github.com/datawhalechina/deep-learning-notes/pull/26)
- DEP: Update dependency transformers to >=5.12.0,<5.13.0 by @renovate[bot] in [#28](https://github.com/datawhalechina/deep-learning-notes/pull/28)
- DEP: Update dependency pytest to >=9.1.0,<9.2.0 by @renovate[bot] in [#29](https://github.com/datawhalechina/deep-learning-notes/pull/29)
- CI: Update actions/checkout action to v7 by @renovate[bot] in [#31](https://github.com/datawhalechina/deep-learning-notes/pull/31)
- DEP: Update dependency scipy to >=1.18.0,<1.19.0 by @renovate[bot] in [#32](https://github.com/datawhalechina/deep-learning-notes/pull/32)
- DEP: Update dependency numpy to >=2.5.0,<2.6.0 by @renovate[bot] in [#33](https://github.com/datawhalechina/deep-learning-notes/pull/33)
- CI: Update actions/cache action to v6 by @renovate[bot] in [#34](https://github.com/datawhalechina/deep-learning-notes/pull/34)
- DEP: Update dependency pillow to >=12.3.0,<12.4.0 by @renovate[bot] in [#35](https://github.com/datawhalechina/deep-learning-notes/pull/35)
- DEP: Update dependency transformers to >=5.13.0,<5.14.0 by @renovate[bot] in [#38](https://github.com/datawhalechina/deep-learning-notes/pull/38)
- DEP: Update dependency opencv-contrib-python to v5 by @renovate[bot] in [#36](https://github.com/datawhalechina/deep-learning-notes/pull/36)
- DEP: Update dependency diffusers to >=0.39.0,<0.40.0 by @renovate[bot] in [#37](https://github.com/datawhalechina/deep-learning-notes/pull/37)
- DEP: Update dependency regex to >=2026.7.10,<2026.8.0 by @renovate[bot] in [#42](https://github.com/datawhalechina/deep-learning-notes/pull/42)

> [!NOTE]
> This project continues to be maintained in both **Chinese** and **English** through a Quarto-based structure, as an open and continuously growing collection of deep learning study notes.

**Full Changelog**: [https://github.com/datawhalechina/deep-learning-notes/compare/v2026.06.11...v2026.07.21](https://github.com/datawhalechina/deep-learning-notes/compare/v2026.06.11...v2026.07.21)

## June 2026 Release

This release significantly expands the project with new chapters on optimization algorithms, Vision Transformers (ViT), and additional PyTorch fundamentals. The accompanying `dnnl` library has also been extended with new neural network components and model implementations.

### New Notebooks

#### Chapter 3: Multi-Layer Perceptron: From Single Layer to Deep Nonlinear Modeling

- 3.1 From Linear Classifiers to MLPs: Why We Need Hidden Layers
- 3.2 Activation Functions: Adding Nonlinearity to Neural Networks
- 3.3 Softmax and Cross Entropy: From Logits to Classification Loss
- 3.4 Forward and Backward Propagation of Linear Layers
- 3.5 Building a Complete MLP with NumPy
- 3.6 Train MLP on MNIST with NumPy
- 3.7 Backward Propagation Check: Using Numerical Gradients to Verify Handwritten Backward
- 3.8 Reimplementing MLP with PyTorch nn.Module

#### Chapter 4: Optimization Algorithms: How Neural Networks Update Parameters

- 4.1 From Gradient Descent to SGD
- 4.2 Momentum and Nesterov Momentum
- 4.3 Adagrad: Adapting the Learning Rate for Each Parameter
- 4.4 RMSprop and Adadelta: Fixing Adagrad's Learning-rate Decay
- 4.5 Adam: Combining Momentum and Adaptive Scaling
- 4.6 AdamW: Decoupling Weight Decay from Adam
- 4.7 Muon: Orthogonalizing Matrix Updates
- 4.8 Optimizer Map: When to Use Which Optimization Algorithm
- 4.9 Learning Rate Schedulers: How the Learning Rate Changes During Training

#### Chapter 11: Vision Transformer: From Image Classification to Visual Sequence Modeling

- 11.1 From CNN to Vision Transformer: Treating Images as Sequences
- 11.2 Patch Embedding: Cutting Images into Tokens
- 11.3 Class Token and Positional Embedding: Letting a Sequence Represent the Whole Image
- 11.4 ViT Encoder: Letting Patch Tokens Exchange Information
- 11.5 ViT Backbone: Pretraining and Fine-tuning

### `dnnl` Package Updates

- Added NumPy-based implementations of common neural network building blocks, including linear layers, activation functions, loss functions, normalization layers, and optimizers.
- Added a complete NumPy MLP implementation with forward propagation, backpropagation, gradient checking, and MNIST training examples.
- Added Vision Transformer (ViT) components, including patch embedding, class tokens, positional embeddings, Transformer encoders, and classification heads.
- Expanded Transformer-related modules and improved interoperability between educational examples and reusable library code.
- Added optimizer implementations including SGD, momentum, Nesterov momentum, Adagrad, RMSprop, Adam, AdamW, and Muon.
- Added learning rate scheduler and optimizer-related utilities.
- Improved package organization and documentation across neural network, optimization, and vision-related modules.
- Expanded test coverage and examples for newly introduced models and optimization algorithms.
- Updated package metadata, dependencies, CI workflows, and development tooling.

### CI Updates

- Migrated GitHub Actions workflows to use GitHub Artifact Attestations for build provenance and artifact verification.
- Replaced Quarto `_freeze` caching with GitHub Actions cache to reduce repository size and improve CI performance.
- Improved workflow reliability and build reproducibility across documentation and package pipelines.

### Merged Pull Requests

- DEP: Bump numpy from 2.4.5 to 2.4.6 by @dependabot[bot] in [#5](https://github.com/datawhalechina/deep-learning-notes/pull/5)
- DEP: Update transformers requirement from ~=5.8.0 to ~=5.9.0 by @dependabot[bot] in [#8](https://github.com/datawhalechina/deep-learning-notes/pull/8)
- FIX: Fix view operations for q, k, v in multi-head attention by @kbyy123 in [#11](https://github.com/datawhalechina/deep-learning-notes/pull/11)
- FIX: Fix some typos in decoder explanation by @kbyy123 in [#12](https://github.com/datawhalechina/deep-learning-notes/pull/12)
- [en] DOC: Fix formula rendering issues in ch1.3 based on CN version by @wqpwqp1222 in [#13](https://github.com/datawhalechina/deep-learning-notes/pull/13)
- DEP: Update dependency gdown to >=6.1.0,<6.2.0 by @renovate[bot] in [#16](https://github.com/datawhalechina/deep-learning-notes/pull/16)
- DEP: Update dependency scikit-learn to >=1.9.0,<1.10.0 by @renovate[bot] in [#17](https://github.com/datawhalechina/deep-learning-notes/pull/17)
- FIX: Remove extra 'not' in zero_grad example code for both zh and en versions by @wqpwqp1222 in [#18](https://github.com/datawhalechina/deep-learning-notes/pull/18)
- DEP: Update dependency transformers to >=5.10.1,<5.11.0 by @renovate[bot] in [#19](https://github.com/datawhalechina/deep-learning-notes/pull/19)
- DEP: Update dependency datasets to v5 by @renovate[bot] in [#20](https://github.com/datawhalechina/deep-learning-notes/pull/20)
- DEP: Update dependency diffusers to >=0.38.0,<0.39.0 by @renovate[bot] in [#24](https://github.com/datawhalechina/deep-learning-notes/pull/24)
- DEP: Update dependency transformers to >=5.11.0,<5.12.0 by @renovate[bot] in [#25](https://github.com/datawhalechina/deep-learning-notes/pull/25)

### New Contributors

- @kbyy123 made their first contribution in [#11](https://github.com/datawhalechina/deep-learning-notes/pull/11)
- @wqpwqp1222 made their first contribution in [#13](https://github.com/datawhalechina/deep-learning-notes/pull/13)
- @renovate[bot] made their first contribution in [#16](https://github.com/datawhalechina/deep-learning-notes/pull/16)

> [!NOTE]
> This project continues to be maintained in both **Chinese** and **English** through a Quarto-based structure, as an open and continuously growing collection of deep learning study notes.

**Full Changelog**: [https://github.com/datawhalechina/deep-learning-notes/compare/v2026.05.09...v2026.06.11](https://github.com/datawhalechina/deep-learning-notes/compare/v2026.05.09...v2026.06.11)

### May 2026 Release

This release completes the Attention and Transformers chapter, adds English versions for all Chinese content, improves notebook packaging and formatting, and introduces a rewritten `dnnl` package with tests and CI support.

### New Notebooks

#### Chapter 1: Introduction to Deep Learning

- 1.1 Neural Networks: A Learnable Function

#### Chapter 8: Attention and Transformers: From Fixed-Length Encoding to Dynamic Context Modeling

- 8.1 Bahdanau Attention: From Information Compression to Dynamic Retrieval
- 8.2 Cross-Attention: One Sequence Querying Another Sequence
- 8.3 Self-Attention: Internal Information Interaction within a Sequence
- 8.4 Multi-Head Attention: From Single Perspective to Multiple Perspectives
- 8.5 Positional Encoding: Adding Positional Information to Attention
- 8.6 Transformer Encoder: Stacking Self-Attention Layers
- 8.7 Transformer Decoder: Masked Self-Attention and Cross-Attention
- 8.8 Encoder-Decoder Transformer: Connecting Encoder and Decoder
- 8.9 KV Cache: Why We Don't Recompute the Past During Inference
- 8.10 Three Different Transformer Architectures: Understanding, Generation, and Input-Output Conversion
- 8.11 Hugging Face Transformers API: From Structure to Calls

### Repository and Publishing Updates

- Added English versions for all Chinese content.
- Packaged notebooks now include images.
- Refined page navigation, code output wrapping, Open Graph descriptions, blockquote emphasis, and plaintext code block styling.
- Regular version bump.

### `dnnl` Package Updates

- Completely rewrote `dnnl` around a PyTorch-like API, with module classes under `dnnl.nn` and stateless helpers under `dnnl.nn.functional`.
- Removed the old chapter-based package layout, including `dnnl.ch8`, `dnnl.ch10`, `dnnl.ch13`, and `dnnl.ch14`.
- Reorganized `dnnl` into reusable neural-network components instead of chapter-specific modules.
- Added attention, FlashAttention, positional encoding, Transformer, AE/VAE, diffusion, and UNet-related code.
- Improved attention and Transformer APIs to better align with PyTorch behavior.
- Updated projection bias handling, causal masks, attention weights, and functional interfaces.
- Added unit tests for attention, FlashAttention, AE/VAE, diffusion, Transformer, and PyTorch compatibility checks.
- Added a dedicated GitHub Actions workflow for testing and building `dnnl`.
- Updated the `dnnl` version, package metadata, dependencies, package-specific Ruff configuration.

> [!NOTE]
> This project continues to be maintained in both **Chinese** and **English** through a Quarto-based structure, as an open and continuously growing collection of deep learning study notes.

**Full Changelog**: [https://github.com/datawhalechina/deep-learning-notes/compare/v2026.04.21...v2026.05.09](https://github.com/datawhalechina/deep-learning-notes/compare/v2026.04.21...v2026.05.09)

## April 2026 Release

This first release introduces the initial public version of these notes, covering topics from deep learning fundamentals to modern architectures and generative models.

### New Notebooks

#### Chapter 1: Introduction to Deep Learning

- 1.3 Forward Propagation, Backpropagation, and Computation Graphs

#### Chapter 2: Getting Started with PyTorch

- 2.1 Automatic Differentiation in PyTorch
- 2.2 Gradient Recording and Control in PyTorch

#### Chapter 10: FlashAttention: Efficient Implementation of Attention Mechanism

- 10.1 Why Attention is IO-Bound
- 10.2 10.2 Flash Attention v1: Eliminating the IO Bottleneck in Attention Mechanisms

#### Chapter 12: GAN: Generative Adversarial Networks

- 12.1 GANs: The Basics of Generative Adversarial Networks

#### Chapter 13: VAE: Variational Autoencoders

- 13.1 Autoencoder: Starting with Compression and Reconstruction
- 13.2 VAE: Probabilistic Modeling and the Reparameterization Trick
- 13.3 ELBO: Where Does the VAE's Objective Function Come From?
- 13.4 VAE Training Phenomena and Latent Space Intuition
- 13.5 VAE: Advantages, Limitations, and Future Developments

#### Chapter 14: Diffusion Models: From Denoising to Generation

- 14.1 DDPM: From Denoising to Generation
- 14.2 The Forward Process of DDPM: From Image to Noise
- 14.3 DDPM's Reverse Denoising Process and Training Objective
- 14.4 DDPM Network Architecture and Sampling Process
- 14.5 DDPM from a Variational Perspective: Where Does the ELBO Come From?

#### Chapter 15: CLIP: Multimodal Models Integrating Vision and Language

- 15.1 CLIP: Connecting Images and Language with Contrastive Learning

### Repository and Publishing Updates

- Added GitHub Actions workflows for packaging and publishing Quarto notebooks.
- Updated Giscus configuration.
- Added a `_freeze` folder for caching.

### `dnnl` Package Updates

- Added notes and setup-related updates around the `dnnl` package.
- Added chapter-based implementations in `dnnl` to support examples and code used across individual chapters.
- Added a dedicated GitHub Actions workflow for `dnnl` packaging.

> [!NOTE]
> This project continues to be maintained in both **Chinese** and **English** through a Quarto-based structure, as an open and continuously growing collection of deep learning study notes.

**Full Changelog**: [https://github.com/datawhalechina/deep-learning-notes/commits/v2026.04.21](https://github.com/datawhalechina/deep-learning-notes/commits/v2026.04.21)
