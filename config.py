# base path to YOLO directory
MODEL_PATH = "Yolo"

# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# To count the total number of people (True/False).
People_Counter = True

# Threading ON/OFF.
Thread = True

# Set the threshold value for total violations limit.
Threshold = 10

# Enter the ip camera url (e.g., url = 'http://191.138.0.100:8040/video');
# Set url = 0 for webcam.
url = 'http://192.168.83.148:8080/video' #phone video URL

# Turn ON/OFF the message alert feature.
ALERT = True

# Define the max/min safe distance limits (in pixels) between 2 people.
MAX_DISTANCE = 80
MIN_DISTANCE = 50
