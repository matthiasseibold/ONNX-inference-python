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
ort_session = onnxruntime.InferenceSession("pose_resnet_50_256x256.onnx",
                                           providers=['CPUExecutionProvider'])
                                           # providers=['CUDAExecutionProvider'])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


JOINTS = ['0 - r ankle', '1 - r knee', '2 - r hip', '3 - l hip', '4 - l knee', '5 - l ankle',
          '6 - pelvis', '7 - thorax', '8 - upper neck', '9 - head top', '10 - r wrist',
          '11 - r elbow', '12 - r shoulder', '13 - l shoulder', '14 - l elbow', '15 - l wrist']

ARM_POSE_PAIRS = [[9, 8], [8, 7], [7, 6],  # upper body
                  [6, 2], [2, 1], [1, 0],  # right leg
                  [6, 3], [3, 4], [4, 5],  # left leg
                  [7, 12], [12, 11], [11, 10],  # right arm
                  [7, 13], [13, 14], [14, 15]]  # left arm

get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])

while True:

    _, original_image = cam.read()
    # original_image = cv2.imread("img/steffigraf.jpg")
    image = cv2.resize(original_image, (256, 256), interpolation=cv2.INTER_NEAREST)
    image_plot = cv2.resize(image, (64, 64), interpolation=cv2.INTER_NEAREST)
    image = image / 255

    # normalize image for inference
    x = np.array(image)
    for i in range(3):
        x[i] = (x[i] - mean[i]) / std[i]
    x = x.transpose(-1, 0, 1)
    x = np.reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]])
    x = np.float32(x)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_outs = ort_outs[0][0]

    # draw skeleton lines
    key_points = list(get_keypoints(pose_layers=ort_outs))
    is_joint_plotted = [False for i in range(len(JOINTS))]
    for pose_pair in ARM_POSE_PAIRS:
        from_j, to_j = pose_pair

        from_thr, (from_x_j, from_y_j) = key_points[from_j]
        to_thr, (to_x_j, to_y_j) = key_points[to_j]

        IMG_HEIGHT, IMG_WIDTH, _ = original_image.shape

        from_x_j, to_x_j = from_x_j * IMG_WIDTH / 64, to_x_j * IMG_WIDTH / 64
        from_y_j, to_y_j = from_y_j * IMG_HEIGHT / 64, to_y_j * IMG_HEIGHT / 64

        from_x_j, to_x_j = int(from_x_j), int(to_x_j)
        from_y_j, to_y_j = int(from_y_j), int(to_y_j)

        if from_thr > THRESHOLD and not is_joint_plotted[from_j]:
            # this is a joint
            cv2.ellipse(original_image, (from_x_j, from_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[from_j] = True

        if to_thr > THRESHOLD and not is_joint_plotted[to_j]:
            # this is a joint
            cv2.ellipse(original_image, (to_x_j, to_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[to_j] = True

        if from_thr > THRESHOLD and to_thr > THRESHOLD:
            # this is a joint connection, plot a line
            cv2.line(original_image, (from_x_j, from_y_j), (to_x_j, to_y_j), (255, 74, 0), 3)

    cv2.imshow("RGB Human Body Tracking", original_image)
    k = cv2.waitKey(1)
    if k == 27:  # Esc key to stop
        break
