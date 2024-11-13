# Awesome-Model-Quantization

Collect model quantization related papers, data, repositories



## Papers

<<<<<<< HEAD
### <a name="Stochastic Quantization "></a> Stochastic Quantization 

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
=======
- [Diffusion Model](#diffusion-model)
- [Zero-shot And Data Free Quant](#zero-shot-and-data-free-quant)
>>>>>>> fda9bec066e2039c72d50c04936dda748a2a5a90

### <a name="Diffusion Model"></a> Diffusion Model

- `[CVPR 2024]` TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models, Huang et al. [Arxiv](https://arxiv.org/abs/2311.16503) | [Github](https://modeltc.github.io/TFMQ-DM/)
>TIAR  optimizes the quantization of the Temporal Information Block, minimizing loss of temporal features.
FSC is a calibration strategy that uses different quantization parameters for activations at different time steps, adapting to their range variations.

- `[CVPR 2024]` Towards Accurate Post-training Quantization for Diffusion Models, Wang et al. [Arxiv](https://arxiv.org/abs/2305.18723)  | [Github](https://github.com/ChangyuanWang17/APQ-DM)
>Distribution-aware Quantization adapts quantization to accommodate the significant variations in activation distributions.
Differentiable Search uses a differentiable search algorithm to optimize the importance weights of quantization functions across timesteps.
SRM Principle selects optimal timesteps for informative calibration image generation.


- `[CVPR 2024]`Post-training Quantization on Diffusion Models, Shang et al. [Arxiv](https://arxiv.org/abs/2211.15736)  | [Github](https://github.com/42Shawn/PTQ4DM)
>introduce NDTC , a novel calibration method that samples a set of time steps from a skewed normal distribution and generates calibration samples through the denoising process to enhance the diversity of time steps in the calibration set.

- `[NuerIPS 2023]`PTQD: Accurate Post-Training Quantization for Diffusion Models, He et al. [Arxiv](https://arxiv.org/abs/2305.10657)  | [Github](https://github.com/ziplab/PTQD)
>Correlation Disentanglement separates quantization noise into correlated and uncorrelated parts, allowing for targeted corrections to reduce mean deviation and variance mismatch.
Quantization Noise Correction employs methods to correct both the correlated and uncorrelated parts of the quantization noise, improving the SNR and sample quality.

- `[NuerIPS 2023]`Q-DM: An efficient low-bit quantized diffusion model, Li et al. [Nips](https://papers.nips.cc/paper_files/paper/2023/file/f1ee1cca0721de55bb35cf28ab95e1b4-Paper-Conference.pdf)  
>TaQ addresses the oscillation in activation distributions during training by smoothing fluctuations and introducing precise scaling factors .
NeM tackles the accumulation of quantization errors during multi-step denoising by mimicing the noise estimation capabilities of full-precision models.

- `[ICCV 2023]`Q-Diffusion: Quantizing Diffusion Models, Li et al. [Arxiv](http://arxiv.org/abs/2302.04304)| [Github](https://github.com/Xiuyu-Li/q-diffusion)
>Shortcut-splitting quantization addresses abnormal activation and weight distributions in shortcut layers by performing split quantization on activations and weights before concatenation.

### <a name="Zero-shot And Data Free Quant"></a> Zero-shot And Data Free Quant

- `[CVPR 2020]`ZeroQ: A Novel Zero Shot Quantization Framework, Cai et al. [Arxiv](https://arxiv.org/abs/2001.00281)| [Github](https://github.com/amirgholami/ZeroQ)

- `[CVPR 2021]`Diversifying Sample Generation for Accurate Data-Free Quantization, Zhang et al.[Arxiv](https://arxiv.org/abs/2103.01049)


- `[CVPR 2021]`Zero-shot Adversarial Quantization, Zhang et al. [Arxiv](https://arxiv.org/abs/2103.15263)| [Github](https://github.com/FLHonker/ZAQ-code)

- `[CVPR 2022]`Data-Free Network Compression via Parametric Non-uniform Mixed Precision Quantization, Chikin et al. [CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Chikin_Data-Free_Network_Compression_via_Parametric_Non-Uniform_Mixed_Precision_Quantization_CVPR_2022_paper.pdf)

- `[CVPR 2023]`Hard Sample Matters a Lot in Zero-Shot Quantization, Li et al. [Arxiv](https://arxiv.org/abs/2303.13826)| [Github](https://github.com/lihuantong/HAST)

- `[CVPR 2023]`Adaptive Data-Free Quantization, Qian et al. [Arxiv](https://arxiv.org/abs/2303.06869)| [Github](https://github.com/hfutqian/AdaDFQ)

- `[CVPR 2023]`GENIE: Show Me the Data for Quantization, Jeon et al. [Arxiv](https://arxiv.org/abs/2212.04780)| [Github](https://github.com/SamsungLabs/Genie)

- `[ECCV 2022]`Patch Similarity Aware Data-Free Quantization for Vision Transformers, Li et al. [Arxiv](https://arxiv.org/pdf/2203.02250)| [Github](https://github.com/zkkli/PSAQ-ViT)

- `[NuerIPS 2023]`REx: Data-Free Residual Quantization Error Expansion, Yvinec et al. [Arxiv](https://arxiv.org/pdf/2203.14645)

- `[NuerIPS 2023]`TexQ: Zero-shot Network Quantization with Texture Feature Distribution Calibration, Chen et al. [OpenReview](https://openreview.net/forum?id=r8LYNleLf9)| [Github](https://github.com/dangsingrue/TexQ)

- `[NuerIPS 2022]`ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers, Yao et al. [Arxiv](https://arxiv.org/abs/2206.01861)| [Github](https://github.com/microsoft/DeepSpeed)

- `[AAAI 2024]`Norm Tweaking: High-Performance Low-Bit Quantization of Large Language Models, Li et al. [Arxiv](https://arxiv.org/abs/2309.02784)| [Github](https://github.com/smpanaro/norm-tweaking)

- `[IJCAI 2022]`MultiQuant: Training Once for Multi-bit Quantization of Neural Networks, Xu et al. [IJCAI](https://www.ijcai.org/proceedings/2022/0504.pdf)| [Github](https://github.com/smpanaro/norm-tweaking)

- `[ICLR 2023]`PowerQuant:Automorphism Search For Non-Uniform Quantization, Yvinec et al. [OpenReview](https://openreview.net/forum?id=s1KljJpAukm)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Kai-Liu001/Awesome-Model-Quantization&type=Date)](https://star-history.com/#Kai-Liu001/Awesome-Model-Quantization&Date)
