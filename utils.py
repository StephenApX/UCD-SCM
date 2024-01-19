import numpy as np

def otsu_thres(delta):
    val=np.zeros([256])
    for th in range(256):
        loc1=delta>th
        loc2=delta<=th
        '''
            delta[loc1]=255
            delta[loc2]=0
        '''
        if delta[loc1].size==0:
            mu1=0
            omega1=0
        else:
            mu1=np.mean(delta[loc1])
            omega1=delta[loc1].size/delta.size

        if delta[loc2].size==0:
            mu2=0
            omega2=0
        else:
            mu2=np.mean(delta[loc2])
            omega2=delta[loc2].size/delta.size
        val[th]=omega1*omega2*np.power((mu1-mu2),2)
    loc = np.where(val==np.max(val))
    return loc[0]


def cal_cos_smilarity_float(prev_inte_hire_features, curr_inte_hire_features):
    multi_dotsum_prev_curr = np.sum(prev_inte_hire_features * curr_inte_hire_features, axis=2)[:,:,np.newaxis]
    
    prev_norm = np.linalg.norm(prev_inte_hire_features, axis=2, keepdims=True)
    curr_norm = np.linalg.norm(curr_inte_hire_features, axis=2, keepdims=True)
    multi_dis = prev_norm * curr_norm

    cos_float01 = (multi_dotsum_prev_curr / multi_dis) * 0.5 + 0.5
    return cos_float01

