# Awesome-Model-Quantization

Collect model quantization related papers, data, repositories



## Papers

### Stochastic Quantization 

- `[ICLR 2021]`Degree-Quant: Quantization-Aware Training for Graph Neural Networks,Shyam A. Tailor.[Arxiv](https://arxiv.org/abs/2008.05000)|[Github](https://github.com/camlsys/degree-quant)
>During the training process, some nodes in the graph neural network are randomly protected from quantization, with nodes having higher in-degrees being more likely to be safeguarded, as they are more significantly affected by the reduction in precision.

- `[ICLR 2021]`Training with Quantization Noise for Extreme Model Compression,Angela Fan.[Arxiv](https://arxiv.org/abs/2004.07320)|[Github](https://github.com/facebookresearch/fairseq/tree/main/examples/quant_noise)
>The paper proposes a novel method that introduces quantization noise (Quant-Noise) during the training process to train networks to adapt to extreme compression methods, such as Product Quantization, which typically result in severe approximation errors. This method quantizes only a random subset of weights during each forward pass, allowing other weights to pass gradients without bias. By controlling the amount and form of noise, extreme compression is achieved while maintaining the performance of the original model.


- `[ICLR 2022]`QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization,Xiuying Wei.[Arxiv](https://arxiv.org/abs/2203.05740)|[Github](https://github.com/wimh966/QDrop)
>This paper introduces a novel post-training quantization method called QDROP, aimed at improving the efficiency and accuracy of neural networks under extremely low-bit settings. QDROP achieves this goal by randomly dropping activation quantization during the post-training quantization (PTQ) process. Research shows that properly incorporating activation quantization into PTQ reconstruction can enhance the final model accuracy.


- `[NuerIPS 2020]`Searching for Low-Bit Weights in Quantized Neural Networks,Zhaohui Yang.[Arxiv](https://arxiv.org/abs/2009.08695)|[Github](https://github.com/zhaohui-yang/Binary-Neural-Networks/tree/main/SLB)
>The paper treats the discrete weights in any quantized neural network as searchable variables and uses a differential method for precise search. Specifically, each weight is represented as a probability distribution over a set of discrete values, and these probabilities are optimized during training, with the value having the highest probability being selected to establish the desired quantized network.

- `[NuerIPS 2022]`Leveraging Inter-Layer Dependency for Post -Training Quantization,Changbao Wang.[openview](https://openreview.net/forum?id=L7n7BPTVAr3)
>To alleviate the overfitting issue, NWQ employs Activation Regularization (AR) technology to better control the distribution of activations. To optimize discrete variables, NWQ introduces Annealing Softmax (ASoftmax) and Annealing Mixup (AMixup), which gradually transition the quantized weights and activations from a continuous state to a discrete state.

- `[CVPR 2023]`Bit-Shrinking: Limiting Instantaneous Sharpness for Improving Post-Training Quantization,Chen Lin.[CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Bit-Shrinking_Limiting_Instantaneous_Sharpness_for_Improving_Post-Training_Quantization_CVPR_2023_paper.pdf)
>To smoothen the rough loss surfaces, the paper proposes a method that limits the sharpness term in the loss to reflect the impact of quantization noise. Instead of directly optimizing the target bit-width network, an adaptive bit-width reduction scheduler is designed. This scheduler starts from a higher bit-width and continuously reduces it until it reaches the target bit-width. In this way, the increased sharpness term is kept within an appropriate range.
### Diffusion model

- `[CVPR 2024]` TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models, Huang et al. [Arxiv](https://arxiv.org/abs/2311.16503)|[Github](https://modeltc.github.io/TFMQ-DM/)
> 对时间特征进行了维护 减少了时间步的损失 对每一个时间步使用不同的量化参数


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Kai-Liu001/Awesome-Model-Quantization&type=Date)](https://star-history.com/#Kai-Liu001/Awesome-Model-Quantization&Date)


### test