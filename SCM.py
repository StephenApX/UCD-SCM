import os
import cv2
import numpy as np
from PIL import Image
import argparse
import torch 

from FastSAM.fastsam import FastSAM
try:
    import clip  # for linear_assignment
except (ImportError, AssertionError, AttributeError):
    from ultralytics.yolo.utils.checks import check_requirements
    check_requirements('git+https://github.com/openai/CLIP.git')  # required before installing lap from source
    import clip

from modules import recalibrated_feature_fusion, base_feature_conc
from utils import otsu_thres
from uitls_clip import FastSAMPrompt



def run(params):
    mode = list(params.mode)

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model_sam = FastSAM(str(params.sam_weight_path)) # everything results and  encoder features.
    if 'PSA' in mode:
        model_clip, clip_preprocess = clip.load(str(params.clip_weight_path), device=DEVICE)


    # input data dir.
    img_ext = '.png'
    prev_img_dir = str(params.img_dir_1)
    curr_img_dir = str(params.img_dir_2)

    # result dir.
    # exp_dir = 
    exp_dir = str(params.out_dir)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'bcd_map'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'dis'), exist_ok=True)


    img_names = []
    for f in os.listdir(prev_img_dir):
        if str(f).endswith(img_ext): # png
            img_names.append(f)


    conc_df = None
    res_dict = {}
    for i, img_name in enumerate(img_names):
        print(f'Processing {img_name}', end='\r', flush=True)

        prev_img_path = os.path.join(prev_img_dir, img_name)
        curr_img_path = os.path.join(curr_img_dir, img_name)
        if not os.path.exists(prev_img_path) or not os.path.exists(curr_img_path):
            continue
    
        '''
        Read in input pair image.
        '''
        prev_ori_arr = np.array(cv2.imread(prev_img_path), dtype=np.float32)[:,:,::-1] # trun to rgb
        curr_ori_arr = np.array(cv2.imread(curr_img_path), dtype=np.float32)[:,:,::-1]
        

        '''
        Run FastSAM to acquire everything_results & hierachy feature.
        '''
        prev_everything_results, prev_hier_feats = model_sam(prev_ori_arr, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        curr_everything_results, curr_hier_feats = model_sam(curr_ori_arr, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        

        '''
        Feature fusion (simple concatenation / RFF module) + Calculate cosine-distance.
        '''
        if 'RFF' in mode:
            dm_cons_cossim = recalibrated_feature_fusion(prev_hier_feats, curr_hier_feats)
        else:
            dm_cons_cossim = base_feature_conc(prev_hier_feats, curr_hier_feats)
        dm_cons_cosdis = 1 - dm_cons_cossim
        dm_cons_cosdis = np.clip(dm_cons_cosdis, 0, 1).astype(np.float32).squeeze()


        '''
        Conduct Piecewise Semantic Attention (PSA):
        '''
        if 'PSA' in mode:
            prev_prompt_process = FastSAMPrompt(prev_img_path, prev_everything_results, device=DEVICE)
            prev_bld_score = prev_prompt_process.text_prompt(clip_model=model_clip, preprocess=clip_preprocess)
            curr_prompt_process = FastSAMPrompt(curr_img_path, curr_everything_results, device=DEVICE)
            curr_bld_score = curr_prompt_process.text_prompt(clip_model=model_clip, preprocess=clip_preprocess)

            mean_bld_score = (prev_bld_score + curr_bld_score) / 2. # float.
            conc_bld_score = np.concatenate([np.expand_dims(prev_bld_score, 2), np.expand_dims(curr_bld_score, 2)], 2)
            mean_bld_score = np.max(conc_bld_score, axis=2) # 01

            bld_mask = np.where(mean_bld_score>=0.5, 1, 0).astype(np.float32) # 1
            nonbld_mask = np.where(mean_bld_score<0.5, mean_bld_score, 0).astype(np.float32) # 0-0.5
            strec_nonbld_mask = nonbld_mask * 2 # 0-1
            whole_bld_mask = bld_mask + strec_nonbld_mask

            # mul
            dm_cons_cosdis = np.multiply(dm_cons_cosdis, whole_bld_mask)


        '''
        Merge current CD map.
        '''
        cos_dis_uint8 = (dm_cons_cosdis * 255).astype(np.uint8)
        # collect non-zero results.
        if i == 0:
            conc_df = cos_dis_uint8[cos_dis_uint8 > 0]
        else:
            conc_df = np.concatenate([conc_df, cos_dis_uint8[cos_dis_uint8>0]], axis=0)
        '''
        Save info.
        '''
        res_dict[img_name] = {}
        res_dict[img_name]['dis'] = cos_dis_uint8


        '''
        Save cos-dis map.        
        '''
        out_cos_path = os.path.join(exp_dir, 'dis', img_name)
        cv2.imwrite(out_cos_path, cos_dis_uint8)


    '''
    Global OTSU.
    '''
    thres = otsu_thres(conc_df)
    '''
    Save BCD map.
    '''
    for img_name, img_dict in res_dict.items():
        cos_dis_uint8 = img_dict['dis']
        df_int8 = np.where(cos_dis_uint8>=thres, 255, 0).astype(np.uint8)
        out_png_path = os.path.join(exp_dir, 'bcd_map', img_name)
        cv2.imwrite(out_png_path, df_int8)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Conduct unsupervised change detection on RS image pair based on Segment Change Model (SCM).',
        epilog='Developed by CVEO Team.')
        
    parser.add_argument(
        '-m',
        '--mode',
        help='CD with (PSA) / (RFF) modules.',
        nargs='+',
        default=['PSA', 'RFF'],
        choices=['PSA', 'RFF'])


    parser.add_argument(
        '--sam_weight_path',
        help='path of the FastSAM pt model',
        default='weights/FastSAM_X.pt'
    )
    parser.add_argument(
        '--clip_weight_path',
        help='path of the CLIP pt model',
        default='weights/ViT-B-32.pt'
    )

    parser.add_argument(
        '--img_dir_1',
        help='input dir of images at prev time.',
        default="data/samples_WHU-CD/prev/"
    )
    parser.add_argument(
        '--img_dir_2',
        help='input dir of images at curr time.',
        default="data/samples_WHU-CD/curr/"
    )
    parser.add_argument(
        '-o', '--out_dir',
        help='output CD directory, which consists of bcd_map and dis folders',
        default="results/samples_WHU-CD/"
    )

    parameters = parser.parse_args()
    run(parameters)



