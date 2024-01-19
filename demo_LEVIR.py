from argparse import Namespace
from SCM import run


if __name__ == '__main__':
    
    params = Namespace()
    params.mode = ['PSA', 'RFF']
    params.sam_weight_path = 'weights/FastSAM_X.pt'
    params.clip_weight_path = 'weights/ViT-B-32.pt'
    params.img_dir_1 = "data/samples_LEVIR/A/"
    params.img_dir_2 = "data/samples_LEVIR/B/"
    params.out_dir = 'results/samples_levir/' 
    run(params)







