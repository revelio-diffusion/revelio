---
license: other
language:
- en
pretty_name: LLaVA Pretrain
---


# LLaVA Visual Instruct Pretrain Dataset Card

## Dataset details

**Dataset type:**
LLaVA Visual Instruct Pretrain LCS-558K is a subset of LAION/CC/SBU dataset, filtered with a more balanced concept coverage distribution.
Captions are also associated with [BLIP synthetic caption](https://github.com/salesforce/BLIP#pre-training-datasets-download) for reference.
It is constructed for the pretraining stage for feature alignment in visual instruction tuning.
We aim to build large multimodal towards GPT-4 vision/language capability.

**Dataset date:**
LLaVA Visual Instruct CC3M Pretrain 595K was created in May 2023.

**Dataset structure:**
- `blip_laion_cc_sbu_558k.json` contains the multimodal synthesized conversation from the image-caption pairs, by adding randomly selected instructions like: "Describe this image".  It is used for pretraining in LLaVA.  We use the raw CC-3M caption as the default answer.
- `blip_laion_cc_sbu_558k_meta.json` contains the meta data of the image file name, image URL, synthetic BLIP caption.
- `images.zip` contains all raw images of the filtered subset from LAION/CC/SBU. Important notice: Upon the request from the community, as ~15% images of the original LAION/CC/SBU dataset are no longer accessible, we upload images.zip for better reproducing our work in research community. It should not be used for any other purpose. The use of these images must comply with the LAION/CC/SBU license. This may be taken down when requested by the original LAION/CC/SBU dataset owner or owners of the referenced images.

**Paper or resources for more information:**
https://llava-vl.github.io/

**License:**
Must comply with license of [CC-3M](https://github.com/google-research-datasets/conceptual-captions/blob/master/LICENSE), [BLIP](https://github.com/salesforce/BLIP/blob/main/LICENSE.txt) (if you use their synthetic caption).

CC-3M
The dataset may be freely used for any purpose, although acknowledgement of
Google LLC ("Google") as the data source would be appreciated. The dataset is
provided "AS IS" without any warranty, express or implied. Google disclaims all
liability for any damages, direct or indirect, resulting from the use of the
dataset.


**Where to send questions or comments about the model:**
https://github.com/haotian-liu/LLaVA/issues

## Intended use
**Primary intended uses:**
The primary use of LLaVA is research on large multimodal models and chatbots.

**Primary intended users:**
The primary intended users of the model are researchers and hobbyists in computer vision, natural language processing, machine learning, and artificial intelligence.