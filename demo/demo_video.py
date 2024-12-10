import cv2
import argparse

from demo_model import SGG_Model
import os
from sgg_benchmark.utils.miscellaneous import get_path
import time

# main
def main(args):
    config_path = args.config
    weights = args.weights
    tracking = args.tracking
    rel_conf = args.rel_conf
    box_conf = args.box_conf
    video_path = args.video
    dcs = args.dcs
    save_path = args.save_path

    # this will create and load the model according to the config file
    # please make sure that the path in MODEL.WEIGHT in the config file is correct
    model = SGG_Model(config_path, weights, dcs=dcs, tracking=tracking, rel_conf=rel_conf, box_conf=box_conf)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), fps=frame_rate, frameSize=video_size)

    while True:
        t = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Make prediction
        img, graph = model.predict(frame, visu_type='video')

        # avg_fps.append(fps)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the resulting frame
        cv2.imshow('Bbox detection', img)

        next_frame = (time.time() - t) * frame_rate
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + next_frame)

        if next_frame > 1:
            for i in range(int(next_frame)):
                video_out.write(img)
        else:
            video_out.write(img)

        # when the video is complete, break
        if not ret:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # compute the latency of each component
    print("Latency: \n", model.get_latency())

    # release the video
    video_out.release()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam demo")

    parser.add_argument('--config', default="configs/VG150/baseline/e2e_relation_X_101_32_8_FPN_1x.yaml", type=str, required=True, help='Path to the config file, e.g. config.yml')
    parser.add_argument('--weights', type=str, required=True, help='Path to the weights file, e.g. model.pth')
    parser.add_argument('--tracking', action="store_true", help='Object tracking or not')
    parser.add_argument('--rel_conf', type=float, default=0.05, help='Relation confidence threshold')
    parser.add_argument('--box_conf', type=float, default=0.001, help='Box confidence threshold')
    parser.add_argument('--video', default=".", type=str, help='Path to the video file')
    parser.add_argument('--save_path', default=".", type=str, help='Path to save the output video')

    parser.add_argument('--dcs', type=int, default=100, help='Dynamic Candidate Selection')

    args = parser.parse_args()

    # change all relative paths to absolute
    if not os.path.isabs(args.config):
        args.config = os.path.join(get_path(), args.config)
    if not os.path.isabs(args.weights):
        args.weights = os.path.join(get_path(), args.weights)

    main(args)