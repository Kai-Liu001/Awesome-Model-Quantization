# Awesome-Model-Quantization

Collect model quantization related papers, data, repositories

## Papers

- [Diffusion Model](#diffusion-model)

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


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Kai-Liu001/Awesome-Model-Quantization&type=Date)](https://star-history.com/#Kai-Liu001/Awesome-Model-Quantization&Date)
