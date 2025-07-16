# Efficient Training \& Inference of Large Intelligence Systems

**Mission:** Democratize advanced AI by tackling three bottlenecks—efficient model training, efficient inference, and open-source accessibility—through algorithmic innovation, rigorous theory, and hardware-aware implementation.

## Research Vision

1. **Efficient Model Training**
Design optimizers and sparsity techniques so large models can be trained on commodity GPUs while retaining convergence guarantees.
2. **Efficient Inference**
Develop decoding, quantization, and structured-sparsity methods that push single-GPU generation to thousands of tokens / s for state-of-the-art LLMs.
3. **Open-Source Accessibility**
Release algorithms, reference implementations, and reproducible recipes to lower the entry barrier for researchers and practitioners.

Past leadership at a next-gen hardware-accelerator company informs a holistic *software–hardware co-design* approach and large-scale pre-training infrastructure expertise.

## Past Contributions

| Topic | Key Idea | Impact |
| :-- | :-- | :-- |
| **Online Subspace Descent** (Liang 2024) | Low-rank gradient updates via online PCA | First LLM pre-training on RTX 4090 with lower perplexity than state-of-the-art; provable convergence |
| **Cautious Optimizers** (Liang 2024) | Gradient-direction masking that preserves Hamiltonian structure | Up to 1.47× speed-up on LLaMA \& MAE; merged into 🤗 Transformers; 300 ★ repo |
| **Pixelated Butterfly** (Dao 2021) | Butterfly-based structured sparsity search | 2.5× faster ViT/GPT-2/MLP-Mixer training on ImageNet \& WikiText-103; adopted by Cerebras \& SambaNova |
| **Distributed Lion** (Liu 2024) | Sign-only gradient exchange for Lion optimizer | Slashes bandwidth in multi-node pre-training; powering “training-over-the-Internet” startups |

## Future Directions

1. **Next-Gen Architectures \& Optimizers**
Hybrid Transformer + SSM, Mixture-of-Experts, and few-step optimizers jointly tuned for memory, compute, and communication efficiency.
2. **Inference Acceleration**
Extend industry-proven algorithms—e.g., first 1 k token/s, batch-1 Llama-3/DeepSeek—to new models via speculative decoding, quantization, and sparsity.
3. **Reasoning-Capable Small LMs**
Chain-of-thought distillation, long-context memory efficiency, and agentic training to bring reasoning to sub-1 B-parameter models that run on edge devices (e.g., iPhones). Early 0.5 B parameter Qwen-based model already sees 3 k+ monthly Hugging Face downloads.

## Potential Impact

- **Broader Participation:** Training techniques that fit consumer GPUs enable students, startups, and independent researchers to explore frontier-scale ideas.
- **Lower Environmental Footprint:** Memory, compute, and communication savings translate directly to reduced energy consumption.
- **Industry Acceleration:** Close collaboration with hardware vendors positions these methods for first-class support in future accelerators, giving partners (including Apple) a strategic edge.

**Let’s scale intelligence responsibly—making it faster, greener, and accessible to all.**
