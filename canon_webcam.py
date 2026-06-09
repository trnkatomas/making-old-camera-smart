import gphoto2 as gp
from PIL import Image
import io
import cv2
import numpy as np
import sys
import time
import threading
import queue
from pynput import keyboard
import argparse

# https://github.com/jim-easterbrook/python-gphoto2/issues/13
# https://github.com/jim-easterbrook/python-gphoto2/blob/febff7b8da9fea4e666323b3487c9894f07e33aa/examples/focus-gui.py
# https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
# https://github.com/shantnu/FaceDetect/blob/master/face_detect_cv3.py

context = gp.Context()
camera = gp.Camera()

camera.init()

# Protects all camera operations (libgphoto2 is not thread-safe).
camera_lock = threading.Lock()

# Thread-safe command queues (keyboard callbacks run in a separate thread).
zoom_queue = queue.Queue()
store_queue = queue.Queue()
stats_queue = queue.Queue()

near, far, near_small_step, far_small_step = 2, 6, 0, 4

tracker = cv2.TrackerMOSSE_create()

def set_viewfinder():
    config = camera.get_config()
    param = 'viewfinder'
    widget = gp.check_result(gp.gp_widget_get_child_by_name(config, param))
    gp.gp_widget_set_value(widget, 1)

def focus(closer=True, small_step=False, custom=None, debug=False):
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
    capturetarget_cfg.set_value('Internal RAM')
    imageformat_cfg = cfg.get_child_by_name('imageformat')
    imageformat_cfg.set_value('Small Fine JPEG')
    camera.set_config(cfg)

cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def on_press(key):
    try:
        if key.char == 's':
            store_queue.put_nowait('s')
        elif key.char == 'c':
            stats_queue.put_nowait('c')
    except AttributeError:
        if key == keyboard.Key.page_down:
            zoom_queue.put_nowait('w')
        elif key == keyboard.Key.page_up:
            zoom_queue.put_nowait('t')
        print('special key {0} pressed'.format(key), file=sys.stderr)

def on_release(key):
    if key == keyboard.Key.esc:
        return False

def get_face(image, debug=False, BB=False, debug_wait=False):
    color = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
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

def get_face_frame(image, border_width=5):
    border = np.where(image[:,:,1]==255)
    if border[0].size > 0:
        min_x, min_y = np.argmin(border, axis=1)
        max_x, max_y = np.argmax(border, axis=1)
        return image[border[0][min_x]+border_width:border[0][max_x]-border_width,
                     border[1][min_y]+border_width:border[1][max_y]-border_width, :]

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
    """
    ky1 = np.array(([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]))
    ky2 = np.array(([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]))
    kx1 = np.array(([0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]))
    kx2 = np.array(([0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]))
    g_img = abs(cv2.filter2D(image, cv2.CV_32F, kx1)) + \
            abs(cv2.filter2D(image, cv2.CV_32F, ky1)) + \
            abs(cv2.filter2D(image, cv2.CV_32F, kx2)) + \
            abs(cv2.filter2D(image, cv2.CV_32F, ky2))
    return cv2.boxFilter(g_img * g_img, -1, (window, window), normalize=True)


def SML(image, window_size=3, threshold=7):
    """
    Sum of modified Laplacian according to
    'Depth Map Estimation Using Multi-Focus Imaging'
    by Mendapara
    """
    ky = np.array(([0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [0.0, -1.0, 0.0]))
    kx = np.array(([0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]))
    ml_img = abs(cv2.filter2D(image, cv2.CV_32F, kx)) + \
            abs(cv2.filter2D(image, cv2.CV_32F, ky))
    ret, img_t = cv2.threshold(ml_img, threshold, 0.0, cv2.THRESH_TOZERO)
    return cv2.boxFilter(img_t, -1, (window_size, window_size), normalize=False)

def GLV(image, window_size=3):
    """
    Gray Level Variance according to
    'Depth Map Estimation Using Multi-Focus Imaging'
    by Mendapara
    """
    mean = cv2.boxFilter(image, cv2.CV_32F, (window_size, window_size), normalize=True)
    return (image - mean)**2.0


def tenengrad1(image, window_size=3, threshold=7):
    """
    Tenengrad2b: squared gradient absolute thresholded and
    summed up in each window according to
    'Autofocusing Algorithm Selection in Computer Microscopy'
    by Sun et Al.
    """
    S = cv2.Sobel(image, cv2.CV_32F, 1, 0, 3)**2.0 + \
        cv2.Sobel(image, cv2.CV_32F, 0, 1, 3)**2.0
    ret, dst = cv2.threshold(S, threshold, 0.0, cv2.THRESH_TOZERO)
    return cv2.boxFilter(dst, -1, (window_size, window_size), normalize=False)

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
        with camera_lock:
            focus(False)
    for i in range(130):
        contrast, contrast_all, blur, blur_all = 0, 0, 0, 0
        for j in range(values_per_focus_step):
            with camera_lock:
                im = get_preview_as_image(camera)
            face = get_face(im, debug=True)
            just_face = get_face_frame(face)
            if just_face is not None and just_face.size > 0:
                contrast += get_contrast(just_face)
                contrast_all += get_contrast(face)
                blur += get_bluriness(just_face)
                blur_all += get_bluriness(face)
                with camera_lock:
                    focus()
        print("\t".join([f"{x/values_per_focus_step}" for x in [contrast, contrast_all, blur, blur_all]]),
              file=sys.stderr)

def capture_face(camera, im=None):
    if im is None:
        with camera_lock:
            im = get_preview_as_image(camera, raw=True)
    image = Image.open(im)
    image.load()
    result = get_face(image, debug=True, BB=True)
    if result is None:
        return im, None, None
    face, bb = result
    return im, face, bb


# ---------------------------------------------------------------------------
# Threading helpers
# ---------------------------------------------------------------------------

def _capture_worker(frame_queue, stop_event):
    """Dedicated capture thread: pumps raw JPEG frames as fast as possible.

    Uses a bounded queue with drop-oldest semantics so the main thread always
    gets the most recent frame even when it falls behind momentarily.
    """
    while not stop_event.is_set():
        with camera_lock:
            raw = get_preview_as_image(camera, raw=True)
        if raw is None:
            continue
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put_nowait(raw)


def _cv_worker(cv_queue, cv_result, cv_lock, stop_event):
    """CV thread: face detection without blocking the capture loop.

    Writes the latest (face_image, bounding_box) into cv_result[0] under
    cv_lock.  Designed to lag behind; the main thread feeds it on a
    best-effort basis.
    """
    while not stop_event.is_set():
        try:
            raw_bytes = cv_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
        result = get_face(image, debug=True, BB=True)
        with cv_lock:
            cv_result[0] = result  # (face_img, bb) or None


def _tracker_worker(cv_queue, cv_result, cv_lock, stop_event):
    """CV thread variant for autofocus mode: runs MOSSE tracker.

    Falls back to Haar detection when the tracker loses the subject.
    """
    local_tracker = cv2.TrackerMOSSE_create()
    tracker_initialized = False

    while not stop_event.is_set():
        try:
            raw_bytes = cv_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        frame = cv2.cvtColor(np.array(Image.open(io.BytesIO(raw_bytes))), cv2.COLOR_RGB2BGR)

        if not tracker_initialized:
            image = Image.open(io.BytesIO(raw_bytes))
            image.load()
            result = get_face(image, debug=True, BB=True)
            if result is not None:
                _, bb = result
                local_tracker = cv2.TrackerMOSSE_create()
                local_tracker.init(frame, bb)
                tracker_initialized = True
            continue

        success, box = local_tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            with cv_lock:
                cv_result[0] = (frame, (x, y, w, h))
        else:
            tracker_initialized = False
            with cv_lock:
                cv_result[0] = None


def _handle_keyboard_queues(images, last_raw_bytes):
    """Process keyboard command queues; must only be called from main thread."""
    if not zoom_queue.empty():
        cmd = zoom_queue.get_nowait()
        with camera_lock:
            focus(closer=(cmd == 't'))
        print(f"Focus {'near' if cmd == 't' else 'far'}", file=sys.stderr)

    if not store_queue.empty():
        store_queue.get_nowait()
        path = f'/tmp/face_{images}.jpeg'
        print(f"Storing to {path}", file=sys.stderr)
        with open(path, 'wb') as f:
            f.write(last_raw_bytes)

    if not stats_queue.empty():
        stats_queue.get_nowait()
        # Stats are computed from the most recent CV result in the caller.


def _print_stats(cv_result, cv_lock):
    with cv_lock:
        result = cv_result[0]
    if result is None:
        print("No face detected for stats", file=sys.stderr)
        return
    face, _ = result
    just_face = get_face_frame(face)
    if just_face is not None and just_face.size > 0:
        all_stats = get_picture_stats(face)
        all_stats_face = get_picture_stats(just_face)
        print("Stats:", file=sys.stderr)
        print("all_stats:", '\n\t'.join([str(x) for x in all_stats]), file=sys.stderr)
        print("all_stats_face:", '\n\t'.join([str(x) for x in all_stats_face]), file=sys.stderr)


# ---------------------------------------------------------------------------
# Main capture modes
# ---------------------------------------------------------------------------

def live_capture():
    ensure_capture_to_RAM()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    frame_queue = queue.Queue(maxsize=2)
    cv_queue = queue.Queue(maxsize=1)
    cv_result = [None]  # mutable container so the CV thread can update it
    cv_lock = threading.Lock()
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=_capture_worker, args=(frame_queue, stop_event), daemon=True)
    cv_thread = threading.Thread(
        target=_cv_worker, args=(cv_queue, cv_result, cv_lock, stop_event), daemon=True)
    capture_thread.start()
    cv_thread.start()

    images = 0
    time_start = time.time()
    last_raw_bytes = b''

    try:
        while True:
            raw = frame_queue.get()
            last_raw_bytes = raw.getvalue()
            sys.stdout.buffer.write(last_raw_bytes)
            sys.stdout.flush()
            images += 1

            # Feed the CV thread every 10 frames; non-blocking so capture is never stalled.
            if images % 10 == 0 and not cv_queue.full():
                cv_queue.put_nowait(last_raw_bytes)

            _handle_keyboard_queues(images, last_raw_bytes)

            if not stats_queue.empty():
                stats_queue.get_nowait()
                _print_stats(cv_result, cv_lock)

    except KeyboardInterrupt:
        stop_event.set()
        seconds = time.time() - time_start
        print("Processed images {}".format(images), file=sys.stderr)
        print("This is {:.2f} fps".format(images / seconds), file=sys.stderr)
        sys.stderr.flush()
        sys.exit()


def live_capture_autofocus():
    ensure_capture_to_RAM()

    frame_queue = queue.Queue(maxsize=2)
    cv_queue = queue.Queue(maxsize=1)
    cv_result = [None]
    cv_lock = threading.Lock()
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=_capture_worker, args=(frame_queue, stop_event), daemon=True)
    tracker_thread = threading.Thread(
        target=_tracker_worker, args=(cv_queue, cv_result, cv_lock, stop_event), daemon=True)
    capture_thread.start()
    tracker_thread.start()

    images = 0
    time_start = time.time()

    try:
        while True:
            raw = frame_queue.get()
            raw_bytes = raw.getvalue()
            images += 1

            # Feed tracker every 5 frames; non-blocking.
            if images % 5 == 0 and not cv_queue.full():
                cv_queue.put_nowait(raw_bytes)

            with cv_lock:
                result = cv_result[0]

            if result is not None:
                annotated_frame, bb = result
                sys.stdout.buffer.write(annotated_frame.tobytes())
            else:
                sys.stdout.buffer.write(raw_bytes)
            sys.stdout.flush()

    except KeyboardInterrupt:
        stop_event.set()
        seconds = time.time() - time_start
        print("Processed images {}".format(images), file=sys.stderr)
        print("This is {:.2f} fps".format(images / seconds), file=sys.stderr)
        sys.stderr.flush()
        sys.exit()


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
