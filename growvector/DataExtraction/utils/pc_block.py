#点与点cd>10
import pandas as pd
import numpy as np
from cloudvolume import  CloudVolume
import cv2
import pandas as pd
import os
import math
import torch.nn.functional
import resource
import psutil


def write_ply(point, filename, N):
    file = open(filename, 'w')
    file.writelines("ply\n")
    file.writelines("format ascii 1.0\n")
    file.writelines("element vertex " + str(N) + "\n")
    file.writelines("property float x\n")
    file.writelines("property float y\n")
    file.writelines("property float z\n")
    file.writelines("element face 0\n")
    file.writelines("property list uchar int vertex_indices\n")
    file.writelines("end_header\n")
    for i in range(len(point)):
        p = point[i]
        file.writelines(
            str(float(p[0])) + "\t" + str(float(p[1])) + "\t" + str(float(p[2])) + "\t" + "\n")


def spc(vol, segid, filename_pcd, lx, ly, lz , cx, cy, cz):
    #num = (vol==segid).sum()
    #vol = vol.cpu().numpy()
    temp1 = 0 
    vol = np.where(vol == segid, 255, vol)
    if np.sum(vol == 255) == 0:
        return temp1
    vol = np.where(vol != 255, 0, vol)
    '''
    if num< points:
        sample_factor = points//num + 1
        vol = torch.tensor(vol,dtype = float).to(device)
        vol = vol.unsqueeze(0)
        vol = vol.permute(4,0,1,2,3)
        vol = torch.nn.functional.interpolate(vol, scale_factor=(sample_factor.item(), sample_factor.item(), sample_factor.item()))
        vol = torch.squeeze(vol)
        vol = vol.cpu().numpy()
    '''
    fid1 = open(filename_pcd, 'a')
    for i in range(0, int(lz)*2):
        data_tmp0 = vol[:, :, i]
        if np.sum(data_tmp0 == 255) == 0:
            continue
        data_tmp0 = data_tmp0.astype(np.uint8)
        contours0 = cv2.findContours(data_tmp0, cv2.RETR_LIST,  cv2.CHAIN_APPROX_NONE)
        boundary0 = np.array(contours0[0])
        boundary0 = [y for x in boundary0 for y in x]
        boundary0 = np.array(boundary0).squeeze()
        if(len(boundary0.shape) == 1):
            continue
        temp1 += len(boundary0)
        for n in range(0, len(boundary0)):
            #fid1.write(str((boundary0[n, 1]*16 + cx*4 - lx*16)*sample_factor) + "\t" + str(boundary0[n, 0]*16 + cy*4 - ly*16)*sample_factor)
            #fid1.write("\t" + str((i + cz - lz))*sample_factor)
            fid1.write(str((boundary0[n, 1]+cx-lx)*16) + "\t" + str((boundary0[n, 0]+cy-ly)*16))
            fid1.write("\t" + str((i+cz-lz)*40))
            fid1.write("\n")   
    fid1.close()
    del vol
    return temp1
    

 

def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))


# if __name__ == '__main__':
#     p = psutil.Process()
#     print(p.pid)
#     segment_info = '/braindat/lab/wangcx/fafb/test/save_data'
#     #connect_info = '/braindat/lab/wangcx/fafb/test/score'
#     pc = '/braindat/lab/wangcx/fafb/test/pc'
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     for neuron in os.listdir(segment_info):
#         #neuron = '720575940631555615_save.csv'
#         neuron = '720575940617710656_save.csv'
#         dicname = neuron.strip('_save.csv')
#         dicpath = os.path.join(pc, dicname)
#         try:
#             os.makedirs(dicpath)
#         except:
#             continue
#             #pass
#         df = pd.read_csv(os.path.join(segment_info, neuron),index_col=False)
#         for i in range(0, len(df)):
def get_complete_pc(row, pc_path, vol_ffn):
    temp1 = 0
    segid = int(row[0])
    filename_pcd = os.path.join(pc_path, str(segid) + '.ply')
    bbox = row[1]
    col = row[2]
    cx, cy, cz = float(col[0])//4, float(col[1])//4, float(col[2])//1
    if max(float(bbox[0])//4, float(bbox[1])//4, float(bbox[2]))<20:
        lx, ly, lz = 60, 60, 20
        vol = vol_ffn[int(cx - lx):int(cx + lx), int(cy - ly):int(cy + ly), int(cz - lz):int(cz + lz)]
        temp1 = spc(vol, segid, filename_pcd, lx, ly, lz , cx, cy, cz)
        del vol
    else :
        lx, ly, lz = float(bbox[0])//8 + 100, float(bbox[1])//8 + 100, float(bbox[2])//2 + 50
        sample_factor = 1
        sample_factor += lx*ly*lz//56058412
        if sample_factor>1:
            stepz = lz//sample_factor
            for j in range(0,int(sample_factor*2),2):
                vol_block = vol_ffn[int(cx - lx):int(cx + lx), int(cy - ly):int(cy + ly), int(cz - lz + stepz * j):int(cz - lz + stepz * (j+2) )]
                temp0 = spc(vol_block, segid, filename_pcd, lx, ly, stepz , cx, cy, cz-lz+stepz*(j+1))
                del vol_block
                if temp0==0 and temp1!=0:
                    break
                else :
                    temp1 += temp0
        else:
            vol = vol_ffn[int(cx - lx):int(cx + lx), int(cy - ly):int(cy + ly), int(cz - lz):int(cz + lz)]
            temp1 = spc(vol, segid, filename_pcd, lx, ly, lz , cx, cy, cz)
            del vol
    try:
        with open(filename_pcd, 'r+') as fid_:
            content = fid_.read()
            fid_.seek(0,0)
            fid_.write( 'ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n'.format(temp1) + content)
            fid_.close()
    except:
        print(f'{filename_pcd}filename_pcd cannot be sampled in the block')
        return None

    return filename_pcd
