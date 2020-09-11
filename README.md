# making-old-camera-smart
Old cameras can be repurposed, but they would lack some cool features, this repo tries to fix it

This repo was inspired by numerous articles how to repurpose old camera as a webcam in times of Coronacris.

All of them boils down to 2 lines of code :-) 
```bash
sudo modprobe v4l2loopback exclusive_caps=1 max_buffers=2
gphoto2 --stdout --capture-movie | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video2
```

This repo can be as a drop replacement
```bash
python canon_webcam.py --raw | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video2
```

On top of it, it can detect the face
```bash
python canon_webcam.py --face | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video2
```

And the holy grail and main motivation for the whole effore, focus such that image is allways in focus (even with famous nifty fifty lens)
```bash
python canon_webcam.py --face_autofocus | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video2
```