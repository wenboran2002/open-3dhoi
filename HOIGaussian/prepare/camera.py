import json
import os
import argparse

def get_camera(data_dir):
    smplx_param=json.load(open(os.path.join(data_dir,'smplx_parameters.json')))
    focal=smplx_param['focal']
    princpt=smplx_param['princpt']
    K={'K':[[focal[0],0,princpt[0]],[0,focal[0],princpt[1]],[0,0,1]]}
    I={}
    I['rotation']=[[1,0,0],[0,1,0],[0,0,1]]
    I['translation']=[0,0,0]
    with open(os.path.join(data_dir,'calibration.json'),'w') as f:
        json.dump(K,f)
    with open(os.path.join(data_dir,'extrinsic.json'),'w') as f:
        json.dump(I,f)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing smplx_parameters.json')
    args = parser.parse_args()
    get_camera(args.data_dir)
