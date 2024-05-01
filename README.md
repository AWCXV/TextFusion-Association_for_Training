# TextFusion-Association_for_Training

Before using the coarse-to-fine association mechanism implementation, you need to first download the following pre-trained models and organise them as follow:
- weights/vilt_200k_mlm_itm.ckpt [Google Drive](https://drive.google.com/file/d/1HQLo4auw5NH--dWZm507AN4WAkGRsJRP/view?usp=sharing)
- configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/model_final_280758.pkl [Google Drive](https://drive.google.com/file/d/1y8WQ8Hoolejzd97m17Sk-NHuI06DHENr/view?usp=sharing)
- configs/COCO-PanopticSegmentation/model_final_cafdb1.pkl [Google Drive](https://drive.google.com/file/d/1Q7kFq8iazK_aJ6H9vVEV8pfw93pSADiB/view?usp=sharing)

Next, install the detectron2 platform:
```
cd ..
python -m pip install -e TextFusion-Association_for_Training-main
```

Finally, you are ready to use the vision&text modalities in the "input" folder for generating the association maps that will be saved in the "output" folder.
```
python main_genAssociation.py
```
