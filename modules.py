import numpy as np
import cv2
from utils import cal_cos_smilarity_float


def rf_module(in_feats):
    sque_feats = np.ones(in_feats.shape[2])
    for c in range(in_feats.shape[2]):
        sque_feats[c] = np.mean(in_feats[:,:,c])
    
    for i, ave in np.ndenumerate(sque_feats):
        in_feats[:,:,i[0]] = in_feats[:,:,i[0]] * ave
    exit_feats = in_feats
    return exit_feats


def recalibrated_feature_fusion(prev_hier_feats, curr_hier_feats):
    
    '''
    Feature dimensions extracted from fastsam:
    '''
    # ori
        # level1 (1, 320, 128, 128)
        # level2 (1, 640, 64, 64)
        # level3 (1, 640, 32, 32)
        # level0 (1, 160, 256, 256)
    # reform_prev_hier_feats
        # level0 (1, 160, 256, 256)
        # level1 (1, 320, 128, 128)
        # level2 (1, 640, 64, 64)
        # level3 (1, 640, 32, 32)

    reform_prev_hier_feats, reform_curr_hier_feats = [], []
    # whole format
    for ind in [3,0,1,2]:
    # for ind in [0,1,2]:
        prev_hier_feat_tensor, curr_hier_feat_tensor = prev_hier_feats[ind], curr_hier_feats[ind]
        prev_hier_feat_arr, curr_hier_feat_arr = prev_hier_feat_tensor.cpu().numpy(), curr_hier_feat_tensor.cpu().numpy()
        prev_hier_feat_arr = prev_hier_feat_arr[0].transpose([1,2,0]) # out: channel last
        curr_hier_feat_arr = curr_hier_feat_arr[0].transpose([1,2,0])
        reform_prev_hier_feats.append(prev_hier_feat_arr)
        reform_curr_hier_feats.append(curr_hier_feat_arr)
    

    # stage3
    c3_prev_feats, c3_curr_feats = reform_prev_hier_feats[3], reform_curr_hier_feats[3] # 16,16,640
    c3_prev_feats, c3_curr_feats = rf_module(c3_prev_feats), rf_module(c3_curr_feats) # se
    c3_prev_feats_160 = c3_prev_feats[:,:,list(range(0,c3_prev_feats.shape[2],c3_prev_feats.shape[2]//160))] # 16,16,160
    c3_curr_feats_160 = c3_curr_feats[:,:,list(range(0,c3_curr_feats.shape[2],c3_curr_feats.shape[2]//160))]
    p3_prev_feats = cv2.resize(c3_prev_feats_160, (c3_prev_feats_160.shape[0]*2,c3_prev_feats_160.shape[1]*2), interpolation=cv2.INTER_LINEAR) # 32,32,160
    p3_curr_feats = cv2.resize(c3_curr_feats_160, (c3_curr_feats_160.shape[0]*2,c3_curr_feats_160.shape[1]*2), interpolation=cv2.INTER_LINEAR)

    # stage2
    c2_prev_feats, c2_curr_feats = reform_prev_hier_feats[2], reform_curr_hier_feats[2] # 32,32,640
    c2_prev_feats, c2_curr_feats = rf_module(c2_prev_feats), rf_module(c2_curr_feats) # se
    c2_prev_feats_160 = c2_prev_feats[:,:,list(range(0,c2_prev_feats.shape[2],c2_prev_feats.shape[2]//160))] # 32,32,160
    c2_curr_feats_160 = c2_curr_feats[:,:,list(range(0,c2_curr_feats.shape[2],c2_curr_feats.shape[2]//160))]
    # c2 + p3
    c2_prev_feats_160 = c2_prev_feats_160 + p3_prev_feats # 32,32,160
    c2_curr_feats_160 = c2_curr_feats_160 + p3_curr_feats
    # p2
    p2_prev_feats = cv2.resize(c2_prev_feats_160, (c2_prev_feats_160.shape[0]*2,c2_prev_feats_160.shape[1]*2), interpolation=cv2.INTER_LINEAR) # 64,64,160
    p2_curr_feats = cv2.resize(c2_curr_feats_160, (c2_curr_feats_160.shape[0]*2,c2_curr_feats_160.shape[1]*2), interpolation=cv2.INTER_LINEAR)

    # stage1
    c1_prev_feats, c1_curr_feats = reform_prev_hier_feats[1], reform_curr_hier_feats[1] # 64,64,320
    c1_prev_feats, c1_curr_feats = rf_module(c1_prev_feats), rf_module(c1_curr_feats) # se
    c1_prev_feats_160 = c1_prev_feats[:,:,list(range(0,c1_prev_feats.shape[2],c1_prev_feats.shape[2]//160))] # 64,64,160
    c1_curr_feats_160 = c1_curr_feats[:,:,list(range(0,c1_curr_feats.shape[2],c1_curr_feats.shape[2]//160))]
    # c1 + p2
    c1_prev_feats_160 = c1_prev_feats_160 + p2_prev_feats # 64,64,160
    c1_curr_feats_160 = c1_curr_feats_160 + p2_curr_feats
    


    # s
    s3_prev_128_160 = cv2.resize(c3_prev_feats_160, (256,256), interpolation=cv2.INTER_LINEAR)
    s3_curr_128_160 = cv2.resize(c3_curr_feats_160, (256,256), interpolation=cv2.INTER_LINEAR)
    s2_prev_128_160 = cv2.resize(c2_prev_feats_160, (256,256), interpolation=cv2.INTER_LINEAR)
    s2_curr_128_160 = cv2.resize(c2_curr_feats_160, (256,256), interpolation=cv2.INTER_LINEAR)
    s1_prev_128_160 = cv2.resize(c1_prev_feats_160, (256,256), interpolation=cv2.INTER_LINEAR)
    s1_curr_128_160 = cv2.resize(c1_curr_feats_160, (256,256), interpolation=cv2.INTER_LINEAR)
    # concate s
    cons_prev_feats = np.concatenate([s3_prev_128_160, s2_prev_128_160, s1_prev_128_160], axis=2)
    cons_curr_feats = np.concatenate([s3_curr_128_160, s2_curr_128_160, s1_curr_128_160], axis=2)

    # upsam->dm concate
    cons_prev_feats_512 = np.concatenate([cv2.resize(cons_prev_feats[:,:,:320], (1024,1024), interpolation=cv2.INTER_LINEAR), cv2.resize(cons_prev_feats[:,:,320:], (1024,1024), interpolation=cv2.INTER_LINEAR)], axis=2) 
    cons_curr_feats_512 = np.concatenate([cv2.resize(cons_curr_feats[:,:,:320], (1024,1024), interpolation=cv2.INTER_LINEAR), cv2.resize(cons_curr_feats[:,:,320:], (1024,1024), interpolation=cv2.INTER_LINEAR)], axis=2)
    

    dm_cons_cossim = cal_cos_smilarity_float(cons_prev_feats_512, cons_curr_feats_512)
    return dm_cons_cossim



def base_feature_conc(prev_hier_feats, curr_hier_feats):   

    prev_inte_hire_features, curr_inte_hire_features = None, None

    # feature dimension: shape:  # batch, channel, width, height.
        # level1 (1, 320, 128, 128)
        # level2 (1, 640, 64, 64)
        # level3 (1, 640, 32, 32)
        # level0 (1, 160, 256, 256)

    # parse hire feature and interpolation to same dimension
    for c, feat_pair in enumerate(zip(prev_hier_feats, curr_hier_feats)):
        
        
        prev_hier_feat_tensor, curr_hier_feat_tensor = feat_pair[:]
        prev_hier_feat_arr, curr_hier_feat_arr = prev_hier_feat_tensor.cpu().numpy(), curr_hier_feat_tensor.cpu().numpy()

        prev_hier_feat_arr = prev_hier_feat_arr[0].transpose([1,2,0])
        curr_hier_feat_arr = curr_hier_feat_arr[0].transpose([1,2,0])


        if prev_hier_feat_arr.shape[2] > 320:
            prev_fea_512_0_320 = cv2.resize(prev_hier_feat_arr[:,:,:320], (1024,1024), interpolation=cv2.INTER_LINEAR)
            prev_fea_512_320_640 = cv2.resize(prev_hier_feat_arr[:,:,320:], (1024,1024), interpolation=cv2.INTER_LINEAR)
            prev_fea_512 = np.concatenate([prev_fea_512_0_320, prev_fea_512_320_640], axis=2) 
            curr_fea_512 = np.concatenate([cv2.resize(curr_hier_feat_arr[:,:,:320], (1024,1024), interpolation=cv2.INTER_LINEAR), cv2.resize(curr_hier_feat_arr[:,:,320:], (1024,1024), interpolation=cv2.INTER_LINEAR)], axis=2) 
        else:
            prev_fea_512 = cv2.resize(prev_hier_feat_arr, (1024,1024), interpolation=cv2.INTER_LINEAR)
            curr_fea_512 = cv2.resize(curr_hier_feat_arr, (1024,1024), interpolation=cv2.INTER_LINEAR)


        if c == 0:
            prev_inte_hire_features = prev_fea_512
            curr_inte_hire_features = curr_fea_512
        else:
            prev_inte_hire_features = np.concatenate([prev_inte_hire_features, prev_fea_512], axis=2)
            curr_inte_hire_features = np.concatenate([curr_inte_hire_features, curr_fea_512], axis=2)

    dm_cons_cossim = cal_cos_smilarity_float(prev_inte_hire_features, curr_inte_hire_features)

    return dm_cons_cossim
