import cv2
import argparse

from demo_model import SGG_Model
import os
from sgg_benchmark.utils.miscellaneous import get_path

# main
def main(args):
    config_path = args.config
    dict_file = args.classes
    weights = args.weights
    tracking = args.tracking
    rel_conf = args.rel_conf
    box_conf = args.box_conf
    video_path = args.video

    # this will create and load the model according to the config file
    # please make sure that the path in MODEL.WEIGHT in the config file is correct
    model = SGG_Model(config_path, dict_file, weights, tracking=tracking, rel_conf=rel_conf, box_conf=box_conf)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    avg_fps = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Make prediction
        img, graph, fps = model.predict(frame, visu=True)

        avg_fps.append(fps)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB)

        # Display the resulting frames
        cv2.imshow('Bbox detection', img)
        cv2.imshow('Graph', graph)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Average FPS: ", sum(avg_fps) / len(avg_fps))
    # When everything is done, release the capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam demo")

    parser.add_argument('--config', default="configs/VG150/baseline/e2e_relation_X_101_32_8_FPN_1x.yaml", type=str, required=True, help='Path to the config file, e.g. config.yml')
    parser.add_argument('--weights', type=str, required=True, help='Path to the weights file, e.g. model.pth')
    parser.add_argument('--classes', default="datasets/vg/VG-SGG-dicts-with-attri.json", type=str, required=True, help='Path to the classes dict file, e.g. VG-SGG-dicts.json')
    parser.add_argument('--tracking', action="store_true", help='Object tracking or not')
    parser.add_argument('--rel_conf', type=float, default=0.3, help='Relation confidence threshold')
    parser.add_argument('--box_conf', type=float, default=0.25, help='Box confidence threshold')
    parser.add_argument('--video', default=".", type=str, help='Path to the video file')

    args = parser.parse_args()

    # change all relative paths to absolute
    if not os.path.isabs(args.config):
        args.config = os.path.join(get_path(), args.config)
    if not os.path.isabs(args.weights):
        args.weights = os.path.join(get_path(), args.weights)
    if not os.path.isabs(args.classes):
        args.classes = os.path.join(get_path(), args.classes)

    main(args)