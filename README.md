# Revelio: Interpreting and Leveraging Visual Semantic Information in Diffusion Models [ICCV '25]
Dahye Kim*, Xavier Thomas*, Deepti Ghadiyaram


### 🔗 [Demo Webpage for Visualizations](https://revelio-diffusion.github.io/revelio/)

---

## **About**

We study rich visual semantic information is represented within various layers and denoising timesteps of different diffusion architectures. We uncover monosemantic interpretable features by leveraging k-sparse autoencoders (k-SAE). We substantiate our mechanistic interpretations via transfer learning using light-weight classifiers on off-the-shelf diffusion models' features. On 4 datasets, we demonstrate the effectiveness of diffusion features for representation learning. We provide in-depth analysis of how different diffusion architectures, pre-training datasets, and language model conditioning impacts visual representation granularity, inductive biases, and transfer learning capabilities. Our work is a critical step towards deepening interpretability of black-box diffusion models. 

---

<img src="./assets/main.jpg" alt="Revelio" width="100%">
<img src="./assets/figure2.jpg" alt="Revelio Figure 2" width="100%">

---

## **📁 Repository Structure**

- **`diffc_image_classification/`**  
  *Image Classification Experiments with Diffusion Features*  
  > Example run file: `diffc_image_classification/run.sh`    

  > Please see the [README](./diffc_image_classification/README.md) for more details.

- **`SD-KSAE/`**  
  *Experiments with K-Sparse Autoencoders (K-SAE) on Diffusion Features*
  > Extract features: `python extract_feature.py`
  
  > Train k-SAE: `python train_ksae.py`
  
- **`LLaVA_Diffusion/`**  
  *Setup of LLaVA with Diffusion Features*  
  > For detailed setup instructions, and to run the code, refer to the [LLaVA repository](https://github.com/haotian-liu/LLaVA).

---
