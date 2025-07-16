## Research Statement

#### Kaizhao Liang

### Research Vision

The advancement of artificial intelligence depends critically on addressing three fundamental limitations that constrain the democratization and scalability of AI systems: **efficient model training**, **efficient inference**, and **open-source accessibility**. My research addresses these challenges through algorithmic innovation, theoretical analysis, and practical implementation, with the goal of making advanced AI systems more accessible and computationally sustainable.

### Past Contributions

#### 1. **Breakthrough in Memory-Efficient Training**

My work on **Online Subspace Descent** represents a significant advance in memory-efficient large language model training[^1][^2][^3]. This optimization algorithm reduces memory requirements by leveraging low-rank gradient structures through online PCA instead of computationally expensive singular value decomposition (SVD). The method achieves lower perplexity than state-of-the-art approaches while narrowing the gap with full-rank baselines on LLaMA models ranging from 60M to 7B parameters[^1][^4]. This work provides the first convergence guarantee for arbitrary projection matrix update rules, making it applicable to various optimizers including Adam and Lion[^4].

#### 2. **Novel Optimization Algorithms**

I have pioneered **Cautious Optimizers**, a family of enhanced optimization algorithms that improve training efficiency with minimal code modification[^5][^6][^7]. This approach introduces a masking mechanism that prevents parameter updates when the proposed direction contradicts the current gradient, preserving the Hamiltonian structure of existing optimizers while achieving up to 1.47x speedup in Llama and MAE pretraining[^6][^8]. The theoretical foundation demonstrates that these optimizers retain convergence guarantees under Lyapunov analysis[^5].

#### 3. **Sparse Training Methodologies**

My contributions to **Pixelated Butterfly** demonstrate significant advances in sparse neural network training[^9][^10][^11]. This method optimizes over a continuous superset of sparse matrices using butterfly matrix products, achieving 2.5x faster training on ImageNet and WikiText-103 tasks for Vision Transformers, GPT-2, and MLP-Mixer models without accuracy loss[^10][^11]. The approach addresses the fundamental challenge of searching for optimal sparsity patterns by providing a structured, hardware-efficient solution.

#### 4. **Distributed Training Innovations**

I have developed **Distributed Lion**, an adaptation of the Lion optimizer for distributed training environments[^12][^13][^14]. By leveraging the sign operator in Lion, this method requires only binary or lower-precision vector communication between workers, significantly reducing communication costs while maintaining comparable performance to standard optimizers[^13][^14]. This work is particularly valuable for large-scale model training where communication bandwidth is a bottleneck.

#### 5. **Theoretical Contributions to Adversarial Learning**

My research on the connections between adversarial transferability and knowledge transferability has provided fundamental insights into deep learning robustness[^15][^16]. This work demonstrates that adversarial transferability can serve as a bidirectional indicator of knowledge transferability, contributing to our understanding of how models generalize across domains[^15].

### Future Research Directions

#### 1. **Next-Generation Efficient Training Systems**

Building on my work in memory-efficient training, I plan to develop unified frameworks that simultaneously optimize memory usage, computational efficiency, and communication overhead. This includes extending Online Subspace Descent to multi-modal models and exploring adaptive rank selection mechanisms that dynamically adjust model capacity during training. Future work will focus on developing theoretical guarantees for these adaptive systems and creating practical implementations for emerging hardware architectures.

#### 2. **Adaptive Inference Acceleration**

I aim to develop intelligent inference systems that dynamically allocate computational resources based on input complexity and quality requirements. This research direction will explore the development of routing mechanisms that can efficiently direct different types of queries to appropriately-sized expert models, building on recent advances in mixture-of-experts architectures. The goal is to create systems that maintain high-quality outputs while dramatically reducing computational costs for routine tasks.

#### 3. **Hardware-Software Co-Design for AI Efficiency**

Future research will focus on co-designing algorithms and hardware architectures to maximize AI system efficiency. This includes developing optimization algorithms specifically tailored for dataflow architectures, exploring novel memory hierarchies for large model training, and creating compiler frameworks that can automatically optimize AI workloads across diverse hardware platforms. The objective is to bridge the gap between algorithmic innovation and practical hardware implementation.

#### 4. **Open-Source AI Infrastructure**

I plan to develop comprehensive open-source frameworks that democratize access to efficient AI training and inference tools. This includes creating modular software libraries that allow researchers to easily experiment with different optimization strategies, developing educational resources that explain the theoretical foundations of efficient AI systems, and building benchmarking tools that enable fair comparison of different approaches. The goal is to accelerate community-driven innovation in efficient AI.

#### 5. **Reasoning-Optimized Training Paradigms**

Given the growing importance of reasoning capabilities in AI systems, I will explore training methodologies specifically designed for reasoning-intensive tasks. This includes developing efficient algorithms for chain-of-thought training, creating memory-efficient methods for handling long reasoning sequences, and designing optimization strategies that balance reasoning quality with computational efficiency. The research will focus on avoiding the "overthinking" phenomenon while maintaining high-quality reasoning outputs.

### Impact and Vision

My research program aims to fundamentally transform how AI systems are trained, deployed, and accessed. By addressing the core limitations of computational efficiency and resource accessibility, this work will enable broader participation in AI research and deployment, reduce the environmental impact of AI systems, and accelerate the development of more capable AI technologies.

The combination of theoretical rigor, practical implementation, and open-source accessibility in my research approach ensures that advances in AI efficiency will benefit the entire research community rather than being confined to resource-rich institutions. This vision aligns with the goal of democratizing AI and making advanced capabilities available to researchers and practitioners worldwide.

<div style="text-align: center">⁂</div>

[^1]: https://arxiv.org/abs/2408.12857

[^2]: https://openreview.net/forum?id=P8rTCT6g45

[^3]: https://neurips.cc/virtual/2024/poster/95328

[^4]: https://www.arxiv.org/pdf/2408.12857.pdf

[^5]: https://www.themoonlight.io/fr/review/cautious-optimizers-improving-training-with-one-line-of-code

[^6]: https://arxiv.org/abs/2411.16085

[^7]: https://github.com/kyleliang919/C-Optim

[^8]: https://huggingface.co/papers/2411.16085

[^9]: https://openreview.net/forum?id=Nfl-iXa-y7R

[^10]: https://collaborate.princeton.edu/en/publications/pixelated-butterfly-simple-and-efficient-sparse-training-for-neur

[^11]: https://arxiv.org/abs/2112.00029

[^12]: https://arxiv.org/abs/2404.00438

[^13]: https://openreview.net/pdf?id=wDirCeTIoz

[^14]: https://neurips.cc/virtual/2024/poster/93167

[^15]: https://arxiv.org/abs/2006.14512

[^16]: https://scholar.google.com/citations?user=qKLmNfoAAAAJ

[^17]: https://dblp.org/pid/239/5146

[^18]: https://kaizhao.me

[^19]: https://aclanthology.org/people/k/kaizhao-liang/

[^20]: https://www.linkedin.com/posts/kaizhao-liang-427a42132_nvidias-ai-dreams-unplugged-dual-rack-activity-7248325290580434944-Qs2H

[^21]: https://paperswithcode.com/author/kaizhao-liang

[^22]: https://api.deepai.org/profile/kaizhao-liang

[^23]: https://www.linkedin.com/in/kaizhao-liang-427a42132

[^24]: https://x.com/kyleliang5

[^25]: https://openreview.net/profile?id=~Kaizhao_Liang1

[^26]: https://www.aimodels.fyi/author-profile/Kaizhao Liang-c7d0d684-4a94-42fe-91e5-5cb00a553abd

[^27]: https://x.com/kyleliang5?lang=en

[^28]: https://rosanneliu.com/dlctfs/dlct_220121.pdf

[^29]: https://research.ibm.com/publications/feature-optimization-for-constituent-parsing-via-neural-networks

[^30]: https://par.nsf.gov/biblio/10317987-pixelated-butterfly-simple-efficient-sparse-training-neural-network-models

[^31]: https://github.com/kyleliang919/Online-Subspace-Descent

[^32]: https://scholar.google.es/citations?user=qKLmNfoAAAAJ

[^33]: https://www.themoonlight.io/en/review/memory-efficient-llm-training-with-online-subspace-descent

[^34]: https://dl.acm.org/doi/10.5555/3666122.3668970

[^35]: https://www.aimodels.fyi/papers/arxiv/memory-efficient-llm-training-online-subspace-descent

[^36]: https://dl.acm.org/doi/10.5555/3737916.3739970

[^37]: https://aclanthology.org/Q16-1014.pdf

[^38]: https://openreview.net/forum?id=TIhiFqGOYC

[^39]: https://github.com/kyleliang919

[^40]: https://arxiv.org/html/2503.16419v1

[^41]: https://arxiv.org/pdf/2503.16419.pdf

[^42]: https://iclr.cc/virtual/2025/workshop/23968

[^43]: https://www.linkedin.com/posts/kaizhao-liang-427a42132_now-do-gpt4o-plz-with-open-weights-we-activity-7282828639576961024-pYhu

[^44]: https://neurips.cc/virtual/2024/poster/93662

[^45]: https://arxiv.org/abs/2505.20643

[^46]: https://www.linkedin.com/posts/kaizhao-liang-427a42132_openai-sambanovaai-activity-7250906775384645632-lxf4

[^47]: https://dl.acm.org/doi/10.1145/3712701

[^48]: https://arxiv.org/html/2405.07518v1

[^49]: https://twitter.com/KyleLiang5/with_replies

[^50]: https://www.themoonlight.io/de/review/cautious-optimizers-improving-training-with-one-line-of-code

[^51]: https://arxiv.org/html/2502.16804v1

[^52]: https://www.aimodels.fyi/papers/arxiv/cautious-optimizers-improving-training-one-line-code

[^53]: https://openaccess.thecvf.com/content/WACV2024W/LLVM-AD/papers/Cui_A_Survey_on_Multimodal_Large_Language_Models_for_Autonomous_Driving_WACVW_2024_paper.pdf

[^54]: https://pediamedai.com/team/

[^55]: https://powerdrill.ai/discover/discover-Cautious-Optimizers-Improving-cm40dj76u1fgb0176a66u2hby

[^56]: https://kaizhao.net/publications/cibm2025nexpr.pdf

[^57]: https://dl.acm.org/doi/abs/10.1145/3719290

[^58]: http://kaizhao.me/kaizhao_cv.pdf

[^59]: https://escholarship.org/content/qt8h24z01x/qt8h24z01x.pdf?t=qnl90g

[^60]: https://arxiv.org/abs/2502.12344

[^61]: https://aclanthology.org/P15-5006/

[^62]: https://www.mdpi.com/2079-9268/15/1/15

[^63]: https://www.linkedin.com/posts/kaizhao-liang-427a42132_the-cycle-has-been-completed-httpslnkdin-activity-7288212094166937600-MEQZ

[^64]: https://arxiv.org/html/2403.18702v2

[^65]: https://weiya711.github.io/publications/plarch2023.pdf

[^66]: https://chatpaper.com/de/paper/5010

[^67]: https://dl.acm.org/doi/10.1145/3694715.3695955

[^68]: https://dl.acm.org/doi/10.5555/3737916.3738499
