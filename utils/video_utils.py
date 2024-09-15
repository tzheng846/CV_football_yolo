import cv2

#reads video by storing each frame until "ret" is false
def read_video(video):
    capture = cv2.VideoCapture(video)
    frames =[]
    while(True):
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_frame, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #shape[1] is x and shape[0] is y
    size = (output_frame[0].shape[1],output_frame[0].shape[0])
    out = cv2.VideoWriter(output_path,fourcc,24,size)
    for frame in output_frame:
        out.write(frame)
    out.release()