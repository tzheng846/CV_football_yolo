from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
sys.path.append("../")
from utils import get_box_center,get_box_width

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        #sv has tracker called bytetrack that assigns objects unique id's
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        #going through batch of size 20 to alieviate memory constraints
        batch_size=20
        detections= []
        for i in range(0,len(frames),batch_size):
            batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections+=batch
        return detections
    def get_object_tracks(self, frames, read_from_stub=False,stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb')as f:
                tracks = pickle.load(f)
            return tracks
        detections = self.detect_frames(frames)
        #output holder
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        #overriding goal keeper to be a player object
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inversed = {v:k for k,v in cls_names.items()}

            #convert to supervision's Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #overriding goal keeper to be a player object
            for object_index,class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_index] = cls_names_inversed["player"]

            #track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                #give each index of frame_detection a name
                bounding_box = frame_detection[0].tolist()
                #worded id- players, etc
                cls_id = frame_detection[3]
                #indexed id- 0,1,2,3,etc
                track_id = frame_detection[4]

                
                if cls_id == cls_names_inversed["player"]:
                    tracks["players"][frame_num][track_id] = {"bounding box":bounding_box}
                
                if cls_id == cls_names_inversed["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bounding box":bounding_box}

            for frame_detection in detection_supervision:
                #give each index of frame_detection a name
                bounding_box = frame_detection[0].tolist()
                #worded id- players, etc
                cls_id = frame_detection[3]

                if cls_id == cls_names_inversed["ball"]:
                    tracks["ball"][frame_num][1] = {"bounding box":bounding_box}
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_elisp(self,frame,bounding_box,line_color,track_id=None):
        x_center,y_center = get_box_center(bounding_box)
        width = int(get_box_width(bounding_box))
        cv2.ellipse(
            frame, #image
            center = (x_center,int(bounding_box[3])),#center x and y coordinates
            axes = (width,int(0.3*width)), #major and minor axis (minor set to 30 percent of major)
            angle= 0.0,
            startAngle= -45,
            endAngle=225,
            color= line_color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        #draw the rectangle id tag
        rectangle_width ,rectangle_height= 40,20
        rectange_x1,rectange_x2 = x_center-rectangle_width//2,x_center+rectangle_width//2

        height_buffer = 15
        rectange_y1,rectange_y2 = ((bounding_box[3]-rectangle_height//2)+height_buffer,
                                   (bounding_box[3]+rectangle_height//2)+height_buffer)
        
        if(track_id is not None):
            cv2.rectangle(
                img=frame,
                pt1= (int(rectange_x1),int(rectange_y1)),
                pt2= (int(rectange_x2),int(rectange_y2)),
                color= line_color,
                thickness = -1
                )
            #location of the texts
            text_padding = (12,15)
            text_x1 = rectange_x1 + text_padding[0]
            #if the numbers are too big (3 digits):
            if(track_id>99):
                text_x1 -=10

            cv2.putText(
                frame,
                text=f"{track_id}",
                org= (int(text_x1),int(rectange_y1)+text_padding[1]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0,0,0),
                thickness=2
            )
        return frame
    
    def draw_triangle(self,frame,boundingbox,triangle_color):
        #center point x,y
        x = get_box_center(boundingbox)[0]
        y= int(boundingbox[1])
        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        
        #filling
        cv2.drawContours(frame,[triangle_points],0,triangle_color,-1)
        #line
        cv2.drawContours(frame,[triangle_points],0,triangle_color,2)
        return frame
#36.04
    def draw_annotations(self,video_frames,tracks):
        annotated_video_frames = []
        for frame_index,frame in enumerate(video_frames):
            #so that the original is not edited over
            frame= frame.copy()

            player_dic = tracks["players"][frame_index]
            ball_dic = tracks["ball"][frame_index]
            referee_dic = tracks["referees"][frame_index]

            #annotate the players
            for id,player in player_dic.items():
                frame = self.draw_elisp(frame,player["bounding box"],(0,255,0),id)

            #annotate the referee
            for id,referee in referee_dic.items():
                frame = self.draw_elisp(frame,referee["bounding box"],(0,255,255))

            #annotate the ball
            for id,ball in ball_dic.items():
                frame = self.draw_triangle(frame,ball["bounding box"],(0,255,0))
            annotated_video_frames.append(frame)

        return annotated_video_frames