'''
    To change the input dataset, please scroll down to the end of this file. Change the parameters of the "Infer" function.
'''

import gradio as gr
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import nltk
import torch
import os
import copy
import time
import io
import numpy as np
import re
import cv2

import ipdb

from PIL import Image
from PIL import ImageFilter

from vilt.config import ex
from vilt.modules import ViLTransformerSS

from vilt.modules.objectives import cost_matrix_cosine, ipot
from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer
from torch.autograd import Variable

import numpy as np
import time

import argparse
import glob
import multiprocessing as mp
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"

def setup_cfg(segArgs):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(segArgs.config_file)
    cfg.merge_from_list(segArgs.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = segArgs.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = segArgs.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = segArgs.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="output/segmentation/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    
    parser.add_argument("--MODEL.WEIGHTS", default='configs/COCO-PanopticSegmentation/model_final_cafdb1.pkl')

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def getFinetunedBinaryMap(finalBinaryMap,maskInstancePrePath):
    h, w = finalBinaryMap.shape;
    interestedInstanceRegions = np.zeros((h,w));

    if os.path.exists(maskInstancePrePath) is False:
        os.mkdir(maskInstancePrePath)
    fileNames = os.listdir(maskInstancePrePath);
    for fileName in fileNames:
        maskInstance = cv2.imread(maskInstancePrePath+fileName, 0)/255.0;
        #The interest map obtained from the vilt model is used to compute the common area generated using this mask
        crossMask = finalBinaryMap*maskInstance;

        #Calculate the weight of the intersecting region in relation to this mask.
        coverRate = np.sum(crossMask)/np.sum(maskInstance);

        #Coverage greater than 0.5 Consider this instance important
        if (coverRate > 0.5):
            interestedInstanceRegions += maskInstance;

    #imsave("outputs5InstanceSegmentation/interestedInstanceRegions"+str(index)+".png",interestedInstanceRegions)
    return interestedInstanceRegions;     


@ex.automain
def main(_config):

    #------------------ ViLT load model Begin -----------------
    _config = copy.deepcopy(_config)

    loss_names = {
        "itm": 0,
        "mlm": 0.5,
        "mpp": 0,
        "vqa": 0,
        "imgcls": 0,
        "nlvr2": 0,
        "irtr": 0,
        "arc": 0,
    }
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

    _config.update(
        {
            "loss_names": loss_names,
        }
    )

    model = ViLTransformerSS(_config)
    model.setup("test")
    model.eval()

    #device = "cuda:2" if _config["num_gpus"] > 0 else "cpu"
    device = "cuda:0"
    model.to(device)

    #------------------ ViLT load model End -----------------    

    #------------------ Segmentation load model Begin -----------------
    print("Segmentation Begin...")
    mp.set_start_method("spawn", force=True)
    segArgs = get_parser().parse_args()

    cfg = setup_cfg(segArgs)

    demoSegmentation = VisualizationDemo(cfg)    
    #------------------ Segmentation load model End -----------------

    def Infer(selectedDataset,imageBegin, imageEnd, textBegin, textEnd):
        #generate_YCbCr Channel of source images

        print("------------------------ ViLT Begin ----------------------");

        image_types = ["vis",'ir'];

        
        for image_index in range(imageBegin, imageEnd + 1):
            #Enumerate each image type
            for image_type in image_types:

                for text_index in range(textBegin,textEnd + 1):
                    
                    if (image_type=="ir"):
                        url = "input/"+selectedDataset+"_"+image_type+"/"+str(image_index)+".png"
                    else:
                        url = "input/"+selectedDataset+"_vis/"+str(image_index)+".png"
                    #Path of the text
                    textFile = open("input/"+selectedDataset+"_text/"+str(image_index)+"/"+ str(image_index) + "_"+ str(text_index)+".txt");
                    
                    mp_text = textFile.readline();
                    textFile.close();
                    allWords = mp_text.split(' ');
                    lenAllWords = len(allWords);

                    lenText = min(lenAllWords,30);
                    mp_text = allWords[0];
                    for iT in range(1,lenText):
                        mp_text = mp_text + " " + allWords[iT];
                        
                    image = Image.open(url).convert("RGB")
                    _w, _h = image.size

                    img = pixelbert_transform(size=384)(image)
                    img = img.unsqueeze(0).to(device)
                    batch = {"text": [""], "image": [None]}
                    inferred_token = [mp_text]
                    batch["image"][0] = img

                    selected_token = ""
                    encoded = tokenizer(inferred_token)

                    #What you want to get is a heat map of all the nouns
                    heatmapAllWords = torch.zeros(image.size[1], image.size[0]);

                    #binary region of interest (ROI)
                    binaryInterestedRegions = torch.zeros(image.size[1], image.size[0]);

                    for hidx in range(len(encoded["input_ids"][0][:-1])):
                        if hidx > 0 and hidx < len(encoded["input_ids"][0][:-1]):
                            with torch.no_grad():

                                #selected_token is the traversed word
                                selected_token = tokenizer.convert_ids_to_tokens(
                                    encoded["input_ids"][0][hidx]
                                )
                                #Get the lexicon of the word traversed.
                                tag_pair = nltk.pos_tag([selected_token]);

                                #If it's a noun or the plural of a noun
                                if (tag_pair[0][1]=="NN" or tag_pair[0][1]=="NNS"):

                                    batch["text"] = inferred_token
                                    batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
                                    batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
                                    batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
                                    infer = model(batch)
                                    txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
                                    txt_mask, img_mask = (
                                        infer["text_masks"].bool(),
                                        infer["image_masks"].bool(),
                                    )
                                    for i, _len in enumerate(txt_mask.sum(dim=1)):
                                        txt_mask[i, _len - 1] = False
                                    txt_mask[:, 0] = False
                                    img_mask[:, 0] = False
                                    txt_pad, img_pad = ~txt_mask, ~img_mask

                                    cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
                                    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
                                    cost.masked_fill_(joint_pad, 0)

                                    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
                                        dtype=cost.dtype
                                    )
                                    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
                                        dtype=cost.dtype
                                    )
                                    T = ipot(
                                        cost.detach(),
                                        txt_len,
                                        txt_pad,
                                        img_len,
                                        img_pad,
                                        joint_pad,
                                        0.1,
                                        1000,
                                        1,
                                    )

                                    plan = T[0]
                                    plan_single = plan * len(txt_emb)
                                    cost_ = plan_single.t()

                                    cost_ = cost_[hidx][1:].cpu()

                                    patch_index, (H, W) = infer["patch_index"]
                                    heatmap = torch.zeros(H, W)
                                    for i, pidx in enumerate(patch_index[0]):
                                        h, w = pidx[0].item(), pidx[1].item()
                                        heatmap[h, w] = cost_[i]

                                    heatmap = (heatmap - heatmap.mean()) / heatmap.std()
                                    heatmap = np.clip(heatmap, 1.0, 3.0)
                                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

                                    _w, _h = image.size

                                    overlay = cv2.resize(np.uint8(heatmap * 255),(_w, _h), interpolation=cv2.INTER_NEAREST)
                                    overlay = torch.from_numpy(overlay);

                                    heatmapAllWords = torch.max(overlay,heatmapAllWords);


                                    path_pre = "output/association/"+selectedDataset+"_imageIndex_" + str(image_index)+"_textIndex_"+str(text_index);
                                    if not os.path.exists(path_pre):
                                        os.mkdir(path_pre)
                                    overlay = Image.fromarray(np.uint8(overlay));
                                    overlay.save(path_pre+"/"+image_type+"_"+selected_token+"_heatmapOverlay.png");


                    zeros_torch = torch.zeros(image.size[1], image.size[0]);
                    binaryInterestedRegions = heatmapAllWords.greater(zeros_torch);

                    heatmapAllWords*=255;

                    finalMap = Image.fromarray(np.uint8(heatmapAllWords), "L")
                    finalInterestedMap = Image.fromarray(np.uint8(binaryInterestedRegions*255), "L");

                    if not os.path.exists(path_pre):
                        os.mkdir(path_pre)
                    finalMap.save(path_pre+"/"+image_type+"_heatmapAllWordsOverlay.png");
                    finalInterestedMap.save(path_pre+"/"+image_type+"_BinaryInterestedMap.png");


            #Start merging two modal association maps, i.e., Binary Interested Map
            
            for text_index in range(textBegin,textEnd+1):            
                path_pre = "output/association/"+selectedDataset+"_imageIndex_" + str(image_index)+"_textIndex_"+str(text_index)+"/";
                ir_binaryMap = path_pre + "ir_BinaryInterestedMap.png";
                vis_binaryMap = path_pre + "vis_BinaryInterestedMap.png";
                image_ir = np.asarray(Image.open(ir_binaryMap));
                image_vis = np.asarray(Image.open(vis_binaryMap));
                finalBinaryInterestedMap = np.maximum(image_ir,image_vis);

                #Put the final binary correlation map into a list
                finalBinaryInterestedMap = Image.fromarray(finalBinaryInterestedMap);
                finalBinaryInterestedMap.save(path_pre+"Final_BinaryInterestedMap.png");

                print(path_pre+"Final_BinaryInterestedMap.png");

        #return
        print("------------------------ ViLT End ----------------------");



        print("------------------------ Segmentation Begin ----------------------");
        for image_index in range(imageBegin,imageEnd+1):

            for image_type in image_types:
                path = "input/"+selectedDataset+"_"+image_type+"/"+str(image_index)+".png";
                img = read_image(path, format="BGR")
                h,w,c = img.shape;


                predictions, visualized_output = demoSegmentation.run_on_image(img, image_index, "output/segmentation/"+selectedDataset+"_"+image_type)


            #If the image itself already contains semantic information (there is no input of text), the all-1 image is directly used as the interest map.
            if (textBegin > textEnd):
                cur = 0;            
                for image_type in image_types:  
                    cur += 1;                          
                    # Iterate through each split instance, and associate it with the i FinalBinaryInterestedMap.
                    finalBinaryMap = np.ones((h//2,w//2));
                    
                    if (cur == 1):
                        finetunedBinaryMap = getFinetunedBinaryMap(finalBinaryMap,"output/segmentation/"+selectedDataset+"_"+image_type+"_Output/"+str(image_index)+"/");
                    else:
                        finetunedBinaryMap += getFinetunedBinaryMap(finalBinaryMap,"output/segmentation/"+selectedDataset+"_"+image_type+"_Output/"+str(image_index)+"/");
                finetunedBinaryMap[finetunedBinaryMap>0] = 1;
                finetunedBinaryMap = Image.fromarray(finetunedBinaryMap*255);
                finetunedBinaryMap = finetunedBinaryMap.convert('L');
                os.mkdir("output/association/"+selectedDataset+"_imageIndex_" + str(image_index)+"_textIndex_"+str(0)+"/");
                finetunedBinaryMap.save("output/association/"+selectedDataset+"_imageIndex_" + str(image_index)+"_textIndex_"+str(0)+"/Final_Finetuned_BinaryInterestedMap.png");
                print("output/association/"+selectedDataset+"_imageIndex_" + str(image_index)+"_textIndex_"+str(0)+"/Final_Finetuned_BinaryInterestedMap.png");            
            else:
                for text_index in range(textBegin,textEnd+1):
                    cur = 0;
                    for image_type in image_types:  
                        cur += 1;                          
                        # Iterate through each split instance, and associate it with the i FinalBinaryInterestedMap.
                        finalBinaryMapPath = "output/association/"+selectedDataset+"_imageIndex_" + str(image_index)+"_textIndex_"+str(text_index)+"/Final_BinaryInterestedMap.png";
                        finalBinaryMap = cv2.imread(finalBinaryMapPath,0)/255.0;
                        if (cur == 1):
                            finetunedBinaryMap = getFinetunedBinaryMap(finalBinaryMap,"output/segmentation/"+selectedDataset+"_"+image_type+"_Output/"+str(image_index)+"/");
                        else:
                            finetunedBinaryMap += getFinetunedBinaryMap(finalBinaryMap,"output/segmentation/"+selectedDataset+"_"+image_type+"_Output/"+str(image_index)+"/");
                    finetunedBinaryMap[finetunedBinaryMap>0] = 1;
                    finetunedBinaryMap = Image.fromarray(finetunedBinaryMap*255);
                    finetunedBinaryMap = finetunedBinaryMap.convert('L');
                    finetunedBinaryMap.save("output/association/"+selectedDataset+"_imageIndex_" + str(image_index)+"_textIndex_"+str(text_index)+"/Final_Finetuned_BinaryInterestedMap.png");
                    print("output/association/"+selectedDataset+"_imageIndex_" + str(image_index)+"_textIndex_"+str(text_index)+"/Final_Finetuned_BinaryInterestedMap.png");
        

        #print("Segmentation End...")
        print("------------------------Segmentation End----------------------");  
  

    #dataset_list = ["IVT","MFNet","medical","IVT_LLVIP_200","IVT_LLVIP_2000"]
    dataset_list = ["IVT","MFNet","medical","IVT_LLVIP_250","IVT_LLVIP_2000"]

    #dataset，image index begin&end，textual description begin&end
    Infer(dataset_list[4],imageBegin = 1, imageEnd = 1, textBegin = 1, textEnd = 5);
    #btn.click(Infer, inputs=[dropDownImageScene,dropDownImagePair,textBox], outputs=[fusedImage,lastFusedImage])        

    #----------------Segmentation End-----------------------    

