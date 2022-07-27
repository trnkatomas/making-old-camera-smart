# Making old camera smart
> Old cameras can be repurposed, but they lack some of cool features of the modern ones, this repo tries to fix it.

I was sad seing my old trusty Canon 50D just collecting dust on a shelf. The other day, I tried to connect it with gphoto and I was amazed by the quality of picture it can provide especially with prime a lens. On the other hand, I was disappointed by the lack of ability to automaticaly focus on face which is especially anoying while using shallow depth of field. Hence this project trying to overcome this limitation.

Disclaimer: This repo was inspired by numerous articles how to repurpose old cameras as a webcam (resurfacing in 2020 in time of the first Corona crisis).

First lesson learned, all the tutorials boil down to these 2 lines of code :-) 
```bash
sudo modprobe v4l2loopback exclusive_caps=1 max_buffers=2
gphoto2 --stdout --capture-movie | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video2
```

This repo can be used as a drop in replacement
```bash
python canon_webcam.py --raw | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video2
```

On top of it, it can detect the face (albeit being rather slow)
```bash
python canon_webcam.py --face | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video2
```
N.B. this feature is not working
And the holy grail and main motivation for the whole effort, autofocus so that face is allways in focus (even with famous nifty fifty lens)
```bash
python canon_webcam.py --face_autofocus | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video2
```
