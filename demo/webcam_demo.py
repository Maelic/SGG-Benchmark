import cv2
import argparse

from demo_model import SGG_Model
import os
from sgg_benchmark.utils.miscellaneous import get_path

# main
def main(config_path, dict_file, weights):

    # this will create and load the model according to the config file
    # please make sure that the path in MODEL.WEIGHT in the config file is correct
    model = SGG_Model(config_path, dict_file, weights, tracking=False)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Make prediction
        img, graph = model.predict(frame, visu=True)

        # Display the resulting frames
        cv2.imshow('Bbox detection', img)
        cv2.imshow('Graph', graph)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam demo")

    parser.add_argument('--config', default="configs/VG150/baseline/e2e_relation_X_101_32_8_FPN_1x.yaml", type=str, required=True, help='Path to the config file, e.g. config.yml')
    parser.add_argument('--weights', type=str, required=True, help='Path to the weights file, e.g. model.pth')
    parser.add_argument('--classes', default="datasets/vg/VG-SGG-dicts-with-attri.json", type=str, required=True, help='Path to the classes dict file, e.g. VG-SGG-dicts.json')

    args = parser.parse_args()

    # change all relative paths to absolute
    if not os.path.isabs(args.config):
        args.config = os.path.join(get_path(), args.config)
    if not os.path.isabs(args.weights):
        args.weights = os.path.join(get_path(), args.weights)
    if not os.path.isabs(args.classes):
        args.classes = os.path.join(get_path(), args.classes)

    # Now args.config contains the value of the --config argument
    config_path = args.config
    dict_file = args.classes
    weights = args.weights

    main(config_path, dict_file, weights)