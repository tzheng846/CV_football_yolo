from utils import read_video,save_video
from tracking import Tracker
def main():
    print("Start Main")
    #reading video
    video_frames = read_video('imported videos/test (17).mp4')

    #initialize tracker
    tracker = Tracker("models/cvFootball-Best.pt")
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path="stubs/track_stubs.pkl")
    #draw ouput
    annotated_video_frames = tracker.draw_annotations(video_frames,tracks)

    #save video
    save_video(annotated_video_frames,"output_videos/output_video.avi")

if __name__ == "__main__":
    main() 