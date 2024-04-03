import cv2
import argparse

from demo_model import SGG_Model

# main
def main(config_path, dict_file):

    # this will create and load the model according to the config file
    # please make sure that the path in MODEL.WEIGHT in the config file is correct
    model = SGG_Model(config_path, dict_file, tracking=True)

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

    parser.add_argument('--config', type=str, required=True, help='Path to the config file, e.g. config.yml')
    parser.add_argument('--classes', type=str, required=True, help='Path to the classes dict file, e.g. VG-SGG-dicts.json')

    args = parser.parse_args()

    # Now args.config contains the value of the --config argument
    config_path = args.config
    dict_file = args.classes

    main(config_path, dict_file)