from background_subtraction.modnet_runner import ModNetWebCam

# Ask the user for camera index
cam_index = 0

ckpt = "F:\Omar 3amora\Photo Booth\MODNet\pretrained\modnet_webcam_portrait_matting.ckpt"

# Pass the user-specified cam_index to the class
webcam_processor = ModNetWebCam(ckpt_path=ckpt, cam_index=cam_index)
webcam_processor.start()
