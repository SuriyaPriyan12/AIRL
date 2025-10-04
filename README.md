# AIRL Assignment 

## Q1 - Vision Transformer on CIFAR-10

### How to Run in Colab
1. Open `Q1.ipynb` in Google Colab
2. Run all cells sequentially 
3. Training takes ~2-3 hours on GPU
4. Models are saved automatically as `best_vit_model.pth`


### Results
| Model | Test Accuracy |
|-------|---------------|
| ViT (Optimized) | 75.75% |

### Performance Tricks Used
- Small patch size (4x4) for 32x32 images
- CIFAR-10 specific normalization
- Strong data augmentation
- AdamW optimizer with cosine scheduling
- Enhanced dropout strategy

---

## Q2 - Text-Driven Image Segmentation with SAM 2

### How to Run in Colab
1. Open `Q2.ipynb` in Google Colab
2. Install dependencies (first cell)
3. Load your image or use sample
4. Enter text prompt (e.g., "dog", "car", "person")
5. View segmentation results

### Pipeline Overview


**Steps:**
1. **Text-to-Region**: GroundingDINO detects objects from text description
2. **Box-to-Points**: Convert bounding boxes to center point coordinates  
3. **Segmentation**: SAM 2 generates precise masks from point prompts
4. **Visualization**: Display original image + mask overlay

### Limitations
- **Text ambiguity**: "person" may detect multiple people
- **Detection dependency**: Poor object detection leads to poor segmentation
- **Threshold sensitivity**: Results vary with detection confidence thresholds
- **Computational cost**: Requires two large models (GroundingDINO + SAM 2)
- **Single image**: Pipeline processes one image at a time

### Example Usage
```python
text_prompt = "dog"
mask, points = text_driven_segmentation(input_image, text_prompt)
