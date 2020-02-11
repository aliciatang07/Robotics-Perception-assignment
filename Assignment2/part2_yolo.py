# USAGE
# python part2_yolo.py --image images/baggage_claim.jpg --yolo yolo

# import the necessary packages
import numpy as np
import time
import cv2
import os
import sys 

yolo_dir = 'yolo'
# Input dir and output dir
if(sys.argv[1] == "train"):

    image_dir = 'data/train/left'
    output_dir = 'data/train/est_bbox'
    bbox_coord_dir = 'data/train/bbox_coord'
    sample_list = ['000001', '000002', '000003', '000004', '000005','000006',
    '000007','000008','000009','000010']
    print("#####TRAINING MODE")

if(sys.argv[1] == "test"):

    image_dir = 'data/test/left'
    output_dir = 'data/test/est_bbox'
    bbox_coord_dir = 'data/test/bbox_coord'
    sample_list = ['000011', '000012', '000013', '000014', '000015']
    print("#####TESTING MODE")


# output_path = 'data/train/yolo_result/000001.png'
# output_dir = 'data/train/yolo_result'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(bbox_coord_dir):
    os.makedirs(bbox_coord_dir)
# minimum probability to filter weak detections  =》精度

confidence_th = 0.1

# threshold when applyong non-maxima suppression  =》 reduce overlap boudning box 
threshold = 0.6
# options end
for sample_name in (sample_list):

    image_path = os.path.join(image_dir, sample_name + ".png")
    output_path = os.path.join(output_dir, sample_name + ".png")
    output_file = open(os.path.join(bbox_coord_dir, sample_name + ".txt"),"w")

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([yolo_dir, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # derive the paths to the YOLO weights and model configurationY
    weightsPath = os.path.sep.join([yolo_dir, "yolov3.weights"])
    configPath = os.path.sep.join([yolo_dir, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_th:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_th,
                            threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            line = "{} {:.2f} {:.2f} {:.2f} {:.2f} ".format(LABELS[classIDs[i]], x, y, w, h)
            output_file.write(line + '\n')


            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    # show the output image
    cv2.imshow("Image", image) 
    cv2.imwrite(output_path,image)


