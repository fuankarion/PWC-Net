import sys
import os
import glob
import cv2
import torch
import numpy as np
from numpy import linalg as LA
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
import models
"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""

def rescaleMagnitudeOF(mOfChannel):
    return (mOfChannel+1)*(127.5)

def writeUOFFile(fileTarget, flowData):
    mag = LA.norm(flowData, axis=2)
    uNorm = rescaleMagnitudeOF(np.divide(flowData[...,0], mag))
    vNorm = rescaleMagnitudeOF(np.divide(flowData[...,1], mag))

    magNorm=np.divide(mag, np.max(mag))*255
    uFlow = np.stack((magNorm,uNorm, vNorm, ), axis=-1)

    cv2.normalize(uFlow, uFlow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(fileTarget, uFlow)

def writeMeanSubtractedUOFFile(fileTarget, flowData):
    flowData[...,0] = flowData[...,0] - np.mean(flowData[...,0])
    flowData[...,1] = flowData[...,1] - np.mean(flowData[...,1])

    mag = LA.norm(flowData, axis=2)
    uNorm = rescaleMagnitudeOF(np.divide(flowData[...,0], mag))
    vNorm = rescaleMagnitudeOF(np.divide(flowData[...,1], mag))

    magNorm=np.divide(mag, np.max(mag))
    uFlow = np.stack((magNorm,uNorm, vNorm, ), axis=-1)

    cv2.normalize(uFlow, uFlow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(fileTarget, uFlow)

def writeFlowFile(filename,uv):
	"""
	According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
	Contact: dqsun@cs.brown.edu
	Contact: schar@middlebury.edu
	"""
	TAG_STRING = np.array(202021.25, dtype=np.float32)
	if uv.shape[2] != 2:
		sys.exit("writeFlowFile: flow must have two bands!");
	H = np.array(uv.shape[0], dtype=np.int32)
	W = np.array(uv.shape[1], dtype=np.int32)
	with open(filename, 'wb') as f:
		f.write(TAG_STRING.tobytes())
		f.write(W.tobytes())
		f.write(H.tobytes())
		f.write(uv.tobytes())

def processImages(net, im1_fn, im2_fn, flow_fn):
    im_all = [imread(img) for img in [im1_fn, im2_fn]]
    im_all = [im[:, :, :3] for im in im_all]

    # rescale the image size to be multiples of 64
    divisor = 64.
    H = im_all[0].shape[0]
    W = im_all[0].shape[1]

    H_ = int(ceil(H/divisor) * divisor)
    W_ = int(ceil(W/divisor) * divisor)
    for i in range(len(im_all)):
    	im_all[i] = cv2.resize(im_all[i], (W_, H_))

    for _i, _inputs in enumerate(im_all):
    	im_all[_i] = im_all[_i][:, :, ::-1]
    	im_all[_i] = 1.0 * im_all[_i]/255.0

    	im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
    	im_all[_i] = torch.from_numpy(im_all[_i])
    	im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])
    	im_all[_i] = im_all[_i].float()

    im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)
    net.eval()
    flo = net(im_all)
    flo = flo[0] * 20.0
    flo = flo.cpu().data.numpy()

    # scale the flow back to the input size
    flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) #
    u_ = cv2.resize(flo[:,:,0],(W,H))
    v_ = cv2.resize(flo[:,:,1],(W,H))
    u_ *= W/ float(W_)
    v_ *= H/ float(H_)
    flo = np.dstack((u_,v_))

    writeUOFFile(flow_fn, flo)

def processVideo(tupleArgz):
    videoKey, targetFlow = tupleArgz
    print('process',videoKey)

    #Load Net
    pwc_model_fn = '/Software/PWC-Net/PyTorch/pwc_net.pth.tar'
    net = models.pwc_dc_net(pwc_model_fn)
    net = net.cuda()

    framesDir = os.path.join(framesRoot, videoKey)
    allVideoFrames = glob.glob(framesDir + '/*.jpg')
    allVideoFrames.sort()

    if not os.path.exists(os.path.join(targetFlow, videoKey )):
        os.makedirs(os.path.join(targetFlow, videoKey ))
        print ('created ', os.path.join(targetFlow, videoKey ))

    for frameIdx in range(len(allVideoFrames)-1):
        frame0 = allVideoFrames[frameIdx]
        frame1 = allVideoFrames[frameIdx+1]
        baseName0 = os.path.basename(frame0)

        flowFileFWD = os.path.join(targetFlow, videoKey, baseName0[:-4]+'-fwd.png')
        processImages(net, frame0, frame1, flowFileFWD, flowFileMSFWD)

        flowFileBWD = os.path.join(targetFlow, videoKey, baseName0[:-4]+'-bwd.png')
        processImages(net, frame1, frame0, flowFileBWD, flowFileMSBWD)
    net = None



framesRoot = '/home/jcleon/jcleon/A2D/frames'
flowRoot = '/home/jcleon/jcleon/A2D/uof'

allVideoKeys = os.listdir(framesRoot)
allVideoKeys.sort()
parArgz=[]
for aVideoKey in allVideoKeys:
    tupleArgz=(aVideoKey, framesRoot, flowRoot)
    parArgz.append(tupleArgz)

#DO NOT MOVE THIS IMPORT AND INSTANTIATION, LIKE SERIUOSLY!!!
import multiprocessing as mp
pool = mp.Pool(processes=9)
pool.map(processVideo, parArgz)
