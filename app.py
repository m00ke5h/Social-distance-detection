import config, thread, argparse, imutils, cv2, os, time
from detector import detect_people
from scipy.spatial import distance as dist
import numpy as np

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", "--input", type=str, default="", help="path to (optional) input video file"
)
ap.add_argument(
    "-d",
    "--display",
    type=int,
    default=1,
    help="whether or not output frame should be displayed",
)
args = vars(ap.parse_args())


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
Labels = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolo.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolo.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# if a video path was not supplied, grab a reference to the camera
if not args.get("input", False):
    print("Starting the live stream :")
    vs = cv2.VideoCapture(config.url)
    if config.Thread:
        cap = thread.ThreadingClass(config.url)
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("Starting the video : ")
    vs = cv2.VideoCapture(args["input"])
    if config.Thread:
        cap = thread.ThreadingClass(args["input"])

path_to_video = None

# loop over the frames from the video stream
while True:
    # read the next frame from the file
    if config.Thread:
        frame = cap.read()

    else:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

    # resize the frame and then detect people in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=Labels.index("person"))

    # initialize the set of indexes that violate the max/min social distance limits
    serious = set()

    # ensure there are *at least* two people detections (required in
    # order to compute our pairwise distance maps)
    if len(results) >= 2:
        # extract all centroids from the results and compute the
        # Euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number of pixels
                if D[i, j] < config.MIN_DISTANCE:
                    # update our violation set with the indexes of the centroid pairs
                    serious.add(i)
                    serious.add(j)

    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract the bounding box and centroid coordinates
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        if i in serious:
            color = (0, 0, 255)

        # draw (1) a bounding box around the person and
        # (2) the centroid coordinates of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 2)

    # draw some of the parameters
    Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
    cv2.putText(
        frame,
        Safe_Distance,
        (470, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 0, 0),
        2,
    )
    Threshold = "Threshold limit: {}".format(config.Threshold)
    cv2.putText(
        frame,
        Threshold,
        (470, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 0, 0),
        2,
    )

    # draw the total number of social distancing violations on the output frame
    text = "Total serious violations: {}".format(len(serious))
    cv2.putText(
        frame,
        text,
        (10, frame.shape[0] - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 0, 255),
        2,
    )

    # Alert function
    if len(serious) >= config.Threshold:
        cv2.putText(
            frame,
            "-ALERT: Violations over limit-",
            (10, frame.shape[0] - 80),
            cv2.FONT_HERSHEY_COMPLEX,
            0.60,
            (0, 0, 255),
            2,
        )
        if config.ALERT:
            # Alert everyone using speaker
            os.system('spd-say "Please maintain social distance"')

    # check to see if the output frame should be displayed to our screen
    if args["display"] > 0:
        # show the output frame
        cv2.imshow("Real-Time Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

cv2.destroyAllWindows()
