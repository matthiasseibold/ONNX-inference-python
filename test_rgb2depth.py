"""We use a pretrained PoseNet with ResNet-50 backbone and input size 256x256 from:
https://arxiv.org/pdf/1804.06208.pdf

pretrained model from:
https://drive.google.com/drive/folders/1g_6Hv33FG6rYRVLXx1SZaaHj871THrRW """

import numpy as np
import onnxruntime
from operator import itemgetter
import cv2

# threshold for joint predictions
THRESHOLD = 0.3

# training statistics
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# get sample image
cam = cv2.VideoCapture(0)

# init ONNX inference session
ort_session = onnxruntime.InferenceSession("model-f6b98070.onnx",
                                           providers=['CPUExecutionProvider'])
                                           # providers=['CUDAExecutionProvider'])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])

while True:

    # _, original_image = cam.read()
    original_image = cv2.imread("img/steffigraf.jpg")
    image = cv2.resize(original_image, (384, 384), interpolation=cv2.INTER_NEAREST)

    # normalize image for inference
    x = np.array(image / 255)
    for i in range(3):
        x[i] = (x[i] - mean[i]) / std[i]
    x = x.transpose(-1, 0, 1)
    x = np.reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]])
    x = np.float32(x)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    depth = ort_outs[0][0]

    depth_max = np.max(depth)
    depth_min = np.min(depth)

    # normalize output
    out = (depth - depth_min) / (depth_max - depth_min)

    cv2.imshow("RGB Depth", out)
    k = cv2.waitKey(1)
    if k == 27:  # Esc key to stop
        break
