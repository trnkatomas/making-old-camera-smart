import gphoto2 as gp
from PIL import Image
import io
import cv2
import numpy as np
import sys
import time

# https://github.com/jim-easterbrook/python-gphoto2/issues/13
# https://github.com/jim-easterbrook/python-gphoto2/blob/febff7b8da9fea4e666323b3487c9894f07e33aa/examples/focus-gui.py
# https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
# https://github.com/shantnu/FaceDetect/blob/master/face_detect_cv3.py

context = gp.Context()
camera = gp.Camera()

camera.init()

def set_viewfinder():
    config = camera.get_config()
    param = 'viewfinder'
    widget = gp.check_result(gp.gp_widget_get_child_by_name(config, param))
    gp.gp_widget_set_value(widget, 1)

#def focus(closer=True, near=2, far=6, custom=None):
def focus(closer=True, near=0, far=6, custom=None):
    config = camera.get_config()
    param = 'manualfocusdrive'    
    choice = near if closer else far
    if custom:
        choice = custom
    widget = gp.check_result(gp.gp_widget_get_child_by_name(config, param))
    value = gp.check_result(gp.gp_widget_get_choice(widget, choice))
    gp.gp_widget_set_value(widget, value)
    return gp.gp_camera_set_config(camera, config, context)

def get_value():
    config = camera.get_config()
    param = 'manualfocusdrive'    
    widget = gp.check_result(gp.gp_widget_get_child_by_name(config, param))
    return gp.gp_widget_get_value(widget)

def get_preview_as_image(camera, raw=False):
    OK, camera_file = gp.gp_camera_capture_preview(camera)
    if OK == gp.GP_OK:
        file_data = camera_file.get_data_and_size()
        if raw:
            return io.BytesIO(file_data)
        else:
            image = Image.open(io.BytesIO(file_data))
            image.load()
            return image
        
cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_face(image, debug=False, debug_wait=False):
    color = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )    
    if debug:        
        for (x, y, w, h) in faces:
            cv2.rectangle(color, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if debug_wait:
            print("Found {0} faces!".format(len(faces)))    
            cv2.imshow("Faces found", color)
            cv2.waitKey(0)
        else:
            return color
    for (x, y, w, h) in faces:            
        return color[y:y+h, x:x+w]
    else:
        None
    

def get_bluriness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def eval_frame(debug=False):
    image = get_preview_as_image(camera)
    face = get_face(image)
    if face is not None:
        print(get_bluriness(face))
    if debug:
        Image.fromarray(face).show()


if __name__ == "__main__":
    images = 0
    time_start = time.time()
    while True:
       try:
           # The most simple raw output
           #by = get_preview_as_image(camera, raw=True)           
           #sys.stdout.buffer.write(by.read())

           # output with detected face
           im = get_preview_as_image(camera)
           face = get_face(im, debug=True)
           by = cv2.imencode('.jpg', face)[1]
           sys.stdout.buffer.write(by)
           sys.stdout.flush()
           images += 1
       except KeyboardInterrupt:
           time_end = time.time()
           seconds = time_end - time_start
           print("Processed images {}".format(images), file=sys.stderr)
           print("This is {:.2f}".format(images/seconds), file=sys.stderr)
           sys.stderr.flush()
           sys.exit()
        #time.sleep(0.05)