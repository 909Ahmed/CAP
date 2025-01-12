# Multimodal GPT Training
### Using CLIP and Phi3 Models with LoRA Fine-Tuning  

**Input Modalities**: Text / Audio / Image  
**Output Modality**: Text  

### Overview  
This repository demonstrates a framework for training a multimodal GPT model with the power of CLIP and Phi3 models. 
The pipeline includes pretraining on image-text captions and fine-tuning for image-text question answering. 
The lightweight fine-tuning method LoRA (Low-Rank Adaptation) is employed for efficient parameter updates.

### Training Steps  

#### **Step 1: Pretraining for Image-Text Captions**  
- **Dataset**: [COCO 2017 Train Dataset](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)  
- **CLIP Model**: [CLIP ViT Base Patch 32](https://huggingface.co/openai/clip-vit-base-patch32)  
- **LLM Model**: [Phi3 Model](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)  
- **Hardware Requirements**:  
  - **VRAM**: 24 GB  
  - **Batch Size**: 4  

#### **Step 2: Fine-Tuning for Image-Text Question Answering**  
- **Dataset**: [Instruct150k Dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)  
- **CLIP Model**: [CLIP ViT Base Patch 32](https://huggingface.co/openai/clip-vit-base-patch32)  
- **LLM Model**: [Phi3 Model](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)  
- **Hardware Requirements**:  
  - **VRAM**: 24 GB  
  - **Batch Size**: 16  

---

### Inspirations  
- **[LLava: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)**  
- **[NextGPT: Any-to-Any Multimodal LLM](https://arxiv.org/pdf/2309.05519)**  

---

### Notes  
- Inference code and gradio spaces will be released soon.
