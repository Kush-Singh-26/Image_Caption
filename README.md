# Image_Caption

Image Caption generation. Provide an image and get it's caption.

Clone this repo by running this command :

```sh
git clone "https://github.com/Kush-Singh-26/Image_Caption.git"
cd Image_Caption
```

Install the required libraries :

```sh
pip install -r requirements.txt
```

Install the Model from Hugging Face :

```sh
pip install huggingface_hub

python -c "from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download(repo_id='Kush26/Image_caption_Generator', filename='best_model_improved.pth'), './best_model_improved.pth')"
```

To perform inference :

```sh
python caption_generator.py --image_path example.jpg
```

Replace `example.jpg` with your image path.

---

A live demo can be accessed [here](https://huggingface.co/spaces/Kush26/Caption_Generator).

Trained model is uploaded to [hugging face](https://huggingface.co/Kush26/Image_caption_Generator).

## How does this work?

- A pretrained CNN (RESNET-101) is used to perform feature extraction / encoding.
- Encoded feature vector is send to LSTM for decoding and predict a caption for the image.

- COCO 2014 dataset is used along with [Karpathy's](https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json) train/validation splits.

- Training was done on Google Colab Notebook on T4 GPUs.
- It was done in 2 phases due to limited compute time available.

---

## Example

Image used for inference: ![Image used for inference](image.png)

- **Caption** : `a man riding a wave on top of a surfboard .`
- You can also check it [here](https://kush26-caption-generator.hf.space/?__theme=dark&deep_link=Dn62nraQD6Q).
