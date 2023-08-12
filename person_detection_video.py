import cv2
import datetime
import imutils
import numpy as np
from PIL import Image

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
# Only enable it if you are using OpenVino environment
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
        
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit



def main():
    cap = cv2.VideoCapture(0)
    global num_id
    num_id = 0

    red = [0,0,255]
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        count = 0
        


        for i in np.arange(0, person_detections.shape[2]):
          
            confidence = person_detections[0, 0, i, 2]
         
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue
                else:
                    
                        person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = person_box.astype("int")

                        box_person = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                        
                        
                        result = frame[startY:endY, startX:endX]


                        if result.any():
                            hsvImage = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

                            lowerLimit, upperLimit = get_limits(color=red)

                            mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
                            redd = cv2.bitwise_and(result,result,mask=mask)
                            mask_ = Image.fromarray(mask)
                            
                            cv2.imshow('RED',redd)

                        # cv2.imshow('RED',red)

                

                
                # hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # cv2.imwrite(f"save/{num_id}.png",result)
                # num_id += 1

                # print("start"+str(startX),end="")
                # print("end"+str(endX))

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)


        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        count = 0

        cv2.imshow("Application", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


main()
