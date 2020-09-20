import gphoto2 as gp
from PIL import Image
import io
import cv2
import numpy as np
import sys
import time
from pynput import keyboard
import argparse

# https://github.com/jim-easterbrook/python-gphoto2/issues/13
# https://github.com/jim-easterbrook/python-gphoto2/blob/febff7b8da9fea4e666323b3487c9894f07e33aa/examples/focus-gui.py
# https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
# https://github.com/shantnu/FaceDetect/blob/master/face_detect_cv3.py

context = gp.Context()
camera = gp.Camera()

camera.init()

zoom_queue = []
store_queue = []
stats_queue = []
near, far, near_small_step, far_small_step = 2, 6, 0, 4

def set_viewfinder():
    config = camera.get_config()
    param = 'viewfinder'
    widget = gp.check_result(gp.gp_widget_get_child_by_name(config, param))
    gp.gp_widget_set_value(widget, 1)

#def focus(closer=True, near=2, far=6, custom=None):
def focus(closer=True, small_step=False,
          custom=None, debug=False):
    config = camera.get_config()
    param = 'manualfocusdrive'
    if small_step:
        choice = near_small_step if closer else far_small_step
    else:    
        choice = near if closer else far
    if custom:
        choice = custom
    widget = gp.check_result(gp.gp_widget_get_child_by_name(config, param))
    value = gp.check_result(gp.gp_widget_get_choice(widget, choice))
    if debug:
        print("value: {}".format(value), file=sys.stderr)
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

def ensure_capture_to_RAM():
    cfg = camera.get_config()
    capturetarget_cfg = cfg.get_child_by_name('capturetarget')
    capturetarget = capturetarget_cfg.get_value()
    capturetarget_cfg.set_value('Internal RAM')
    # camera dependent - 'imageformat' is 'imagequality' on some
    imageformat_cfg = cfg.get_child_by_name('imageformat')
    imageformat = imageformat_cfg.get_value()
    imageformat_cfg.set_value('Small Fine JPEG')
    camera.set_config(cfg)
        
cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def on_press(key):
    try:
        # print('alphanumeric key {0} pressed'.format(key.char),file=sys.stderr)
        if key.char == 't':
            zoom_queue.append('t')
        elif key.char == 'w':
            zoom_queue.append('w')
        elif key.char == 's':
            store_queue.append('s')
        elif key.char == 'c':
            stats_queue.append('c')
        # print(zoom_queue, file=sys.stderr)
    except AttributeError:
        pass
        # print('special key {0} pressed'.format(key), file=sys.stderr)

def on_release(key):
    # print('{0} released'.format(key), file=sys.stderr)
    if key == keyboard.Key.esc:
        # Stop listener
        return False

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

def get_face_frame(image, border_width=5):
    # rely on that the only pixel with face are the one created previously by opencv    
    border = np.where(image[:,:,1]==255)
    if border[0].size > 0:
        min_x, min_y = np.argmin(border,axis=1)
        max_x, max_y = np.argmax(border,axis=1)
        return image[border[0][min_x]+border_width:border[0][max_x]-border_width, border[1][min_y]+border_width:border[1][max_y]-border_width, :]  

def get_bluriness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)[:,:,0]
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    max_v = np.max(gray)
    min_v = np.min(gray)
    contrast = (max_v-min_v)/(max_v+min_v)
    return contrast

def get_picture_stats(image):
    contrast = get_contrast(image)
    blur = get_bluriness(image)
    return image.shape, contrast, blur

def eval_frame(debug=False):
    image = get_preview_as_image(camera)
    face = get_face(image)
    if face is not None:
        print(get_bluriness(face))
    if debug:
        Image.fromarray(face).show()

def make_sharpness_graph(values_per_focus_step=5):
    time.sleep(3)
    for i in range(20):
            focus(False)            
    for i in range(130):  
        contrast, contrast_all, blur, blur_all = 0,0,0,0
        for j in range(values_per_focus_step):
            im = get_preview_as_image(camera)
            face = get_face(im, debug=True)
            just_face = get_face_frame(face)
            if just_face is not None and just_face.size > 0:
                contrast += get_contrast(just_face)
                contrast_all += get_contrast(face)
                blur += get_bluriness(just_face)
                blur_all += get_bluriness(face)
                focus()
        print("\t".join([f"{x/values_per_focus_step}" for x in [contrast,contrast_all,blur,blur_all]]),
                file=sys.stderr)                

# blur > 50, contrast > 0.97
def live_capture():
    images = 0
    time_start = time.time()
    ensure_capture_to_RAM()
    # ...or, in a non-blocking fashion:
    listener = keyboard.Listener(
      on_press=on_press,
      on_release=on_release)
    listener.start()
    last_face_size = [0, 0]

    while True:
        try:            
            if images % 10 == 0:
                # output with detected face            
                im = get_preview_as_image(camera, raw=True)
                image = Image.open(im)
                image.load()
                face = get_face(image, debug=True)                
                sys.stdout.buffer.write(im.getvalue())
                sys.stdout.flush()
            else:
                # The most simple raw output
                by = get_preview_as_image(camera, raw=True)  
                if by:
                    sys.stdout.buffer.write(by.getvalue())
                    sys.stdout.flush()
            images += 1            
            if zoom_queue:
                print("som tu", file=sys.stderr)
                command = zoom_queue.pop()
                if command == 't':
                        focus()
                elif command == 'w':
                        focus(False)                                
            if store_queue:
                store_queue.pop()
                print(f"storring to /tmp/face_{images}.jpeg", file=sys.stderr)
                with open(f'/tmp/face_{images}.jpeg','wb') as ofile:
                    ofile.write(by.getvalue())
            if stats_queue:
                stats_queue.pop()
                just_face = get_face_frame(face)
                if just_face is not None and just_face.size > 0:
                    contrast = get_contrast(just_face)
                    contrast_all = get_contrast(face)
                    blur = get_bluriness(just_face)
                    blur_all = get_bluriness(face)
                    print(f"Stats:\n{just_face.shape}\ncontrast:\t{contrast}\tcontrast_all:\t{contrast_all}\nblur:\t{blur}\tblur_all:\t{blur_all}", file=sys.stderr)
        except KeyboardInterrupt:
            time_end = time.time()
            seconds = time_end - time_start
            print("Processed images {}".format(images), file=sys.stderr)
            print("This is {:.2f}".format(images/seconds), file=sys.stderr)
            sys.stderr.flush()
            sys.exit()
            #time.sleep(0.05)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-l", action="store_true")
    arg_parser.add_argument("-m", action="store_true")
    arguments = arg_parser.parse_args()
    if arguments.l:
        live_capture()
    if arguments.m:
        make_sharpness_graph()


