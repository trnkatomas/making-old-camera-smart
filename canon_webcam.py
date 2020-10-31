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

#tracker = cv2.TrackerCSRT_create()
#tracker = cv2.TrackerKCF_create()
tracker = cv2.TrackerMOSSE_create()

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
        if key.char == 's':
            store_queue.append('s')
        elif key.char == 'c':
            stats_queue.append('c')
    except AttributeError:
        if key == keyboard.Key.page_down:
            zoom_queue.append('w')
        elif key == keyboard.Key.page_up:
            zoom_queue.append('t')
        print('special key {0} pressed'.format(key), file=sys.stderr)

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

def get_face(image, debug=False, BB=False, debug_wait=False):
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
            if BB:
                return color, (x, y, w, h)
            return color
    for (x, y, w, h) in faces:  
        img = color[y:y+h, x:x+w]
        if BB:
            return img, (x, y, w, h)
        return img
        
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
    max_v = np.max(gray)
    min_v = np.min(gray)
    contrast = (max_v-min_v)/(max_v+min_v)
    return contrast

def CMSL(image, window=3):
    """
    Contrast Measure based on squared Laplacian according to
    'Robust Automatic Focus Algorithm for Low Contrast Images
    Using a New Contrast Measure'
    by Xu et Al. doi:10.3390/s110908281
    window: window size= window X window"""
    ky1 = np.array(([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]))
    ky2 = np.array(([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]))
    kx1 = np.array(([0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]))
    kx2 = np.array(([0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]))
    g_img = abs(cv2.filter2D(image, cv2.CV_32F, kx1)) + \
            abs(cv2.filter2D(image, cv2.CV_32F, ky1)) + \
            abs(cv2.filter2D(image, cv2.CV_32F, kx2)) + \
            abs(cv2.filter2D(image, cv2.CV_32F, ky2))
    return cv2.boxFilter(
                            g_img * g_img,
                            -1,
                            (window, window),
                            normalize=True)


def SML(image, window_size=3, threshold=7):
    """
    Sum of modified Laplacian according to
    'Depth Map Estimation Using Multi-Focus Imaging'
    by Mendapara
    """
    # kernels in x- and y -direction for Laplacian
    ky = np.array(([0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [0.0, -1.0, 0.0]))
    kx = np.array(([0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]))
    # add absoulte of image convolved with kx to absolute
    # of image convolved with ky (modified laplacian)
    ml_img = abs(cv2.filter2D(image, cv2.CV_32F, kx)) + \
            abs(cv2.filter2D(image, cv2.CV_32F, ky))
    # sum up all values that are bigger than threshold in window
    ret, img_t = cv2.threshold(ml_img, threshold, 0.0, cv2.THRESH_TOZERO)
    return cv2.boxFilter(img_t,-1, (window_size, window_size), normalize=False)

def GLV(image, window_size=3):
    """
    Gray Level Variance according to
    'Depth Map Estimation Using Multi-Focus Imaging'
    by Mendapara
    """
    # calculate mean for each window
    mean = cv2.boxFilter(image,
                         cv2.CV_32F,
                         (window_size, window_size),
                         normalize=True)
    # return variance=(img[x,y]-mean[x,y])^2
    return (image - mean)**2.0


def tenengrad1(image, window_size=3, threshold=7):
    """
    Tenengrad2b: squared gradient absolute thresholded and 
    summed up in each window
    according to
    'Autofocusing Algorithm Selection in Computer Microscopy'
    by Sun et Al.
    """
    # calculate gradient magnitude:
    S = cv2.Sobel(image, cv2.CV_32F, 1, 0, 3)**2.0 + \
        cv2.Sobel(image, cv2.CV_32F, 0, 1, 3)**2.0
    # threshold image
    ret, dst = cv2.threshold(S, threshold, 0.0, cv2.THRESH_TOZERO)
    # return thresholded image summed up in each window:
    return cv2.boxFilter(dst,
                         -1,
                         (window_size, window_size),
                         normalize=False)

def jaehne(image, window_size=3):
    """Only implemented for window_size 3 or 5 according to
    'Entwicklung einer fokusbasierenden Hoehenmessung mit
    dem "Depth from Focus"-Verfahren' by Dunck
    """
    if window_size == 3:
        kernel = np.array([
                        [1.0, 2.0, 1.0],
                        [2.0, 4.0, 2.0],
                        [1.0, 2.0, 1.0]])
        sum = 16
    elif window_size == 5:
        kernel = np.array([
                        [1.0, 4.0, 6.0, 4.0, 1.0],
                        [4.0, 16.0, 24.0, 16.0, 4.0],
                        [6.0, 24.0, 36.0, 24.0, 6.0],
                        [4.0, 16.0, 24.0, 16.0, 4.0],
                        [1.0, 4.0, 6.0, 4.0, 1.0]])
        sum = 256
    img_t = (image - cv2.filter2D(image, cv2.CV_32F, kernel) / sum)**2
    return cv2.filter2D(img_t, -1, kernel) / sum

def get_picture_stats(image):
    contrast = get_contrast(image)
    blur = get_bluriness(image)
    jaehne_st = jaehne(image)
    cmsl_st = CMSL(image)
    tene = tenengrad1(image)
    sml_st = SML(image)
    glv_st = GLV(image)
    return image.shape, contrast, blur, jaehne_st, cmsl_st, tene, sml_st, glv_st

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

def capture_face(camera, im=None, raw=True, debug=True):
    if not im:
        im = get_preview_as_image(camera, raw=True)
    image = Image.open(im)
    image.load()    
    face, bb = get_face(image, debug=True, BB=True)   
    return im, face, bb

def process_stas_queue(stats_queue, face):
    stats_queue.pop()
    just_face = get_face_frame(face)
    if just_face is not None and just_face.size > 0:
        all_stats = get_picture_stats(face)
        all_stats_face = get_picture_stats(just_face)
        print(f"Stats:\n", file=sys.stderr)
        print("all_stats:", '\n\t'.join([str(x) for x in all_stats]), file=sys.stderr)
        print("all_stats_face", '\n\t'.join([str(x) for x in all_stats_face]), file=sys.stderr)

def manual_focus():
    command = zoom_queue.pop()
    if command == 't':
        focus()
        print(f"Zoomed in", file=sys.stderr)
    elif command == 'w':
        focus(False)  
        print(f"Zoomed out", file=sys.stderr)

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
    while True:
        try:            
            if images % 10 == 0:
                # output with detected face            
                last_face = face                      
                im, face, bb = capture_face(camera)                        
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
                manual_focus(zoom_queue)                              
            if store_queue:
                store_queue.pop()
                print(f"storring to /tmp/face_{images}.jpeg", file=sys.stderr)
                with open(f'/tmp/face_{images}.jpeg','wb') as ofile:
                    ofile.write(by.getvalue())
            if stats_queue:
                process_stats_queue(stats_queue, face)
        except KeyboardInterrupt:
            time_end = time.time()
            seconds = time_end - time_start
            print("Processed images {}".format(images), file=sys.stderr)
            print("This is {:.2f}".format(images/seconds), file=sys.stderr)
            sys.stderr.flush()
            sys.exit()
            #time.sleep(0.05)

def live_capture_autofocus():
    images = 0
    tracker_initialized = False
    time_start = time.time()
    ensure_capture_to_RAM()
    im, face, bb = capture_face(camera)
    if bb is not None:
        tracker.init(face, bb)
        tracker_initialized = True

    while True:
        try:
            # output with detected face            
            last_face = face     
            im = get_preview_as_image(camera, raw=True)     
            images += 1 
            if images % 5 == 0:                         
                if not tracker_initialized:
                    im, face, bb = capture_face(camera, im)                        
                    tracker.init(face, bb)                            
                frame = cv2.cvtColor(np.array(Image.open(im)), cv2.COLOR_RGB2BGR)  
                (success, box) = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    print(f"rect {(x, y)}, {(x + w, y + h)}", file=sys.stderr)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
                sys.stdout.buffer.write(frame.tostring())
                sys.stdout.flush()
            else:
                sys.stdout.buffer.write(im.getvalue())
                sys.stdout.flush()
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
    arg_parser.add_argument("-a", action="store_true")
    arguments = arg_parser.parse_args()
    if arguments.l:
        live_capture()
    if arguments.m:
        make_sharpness_graph()
    if arguments.a:
        live_capture_autofocus()


