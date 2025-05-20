import cv2
import argparse

from demo_model import SGG_Model
import os
from sgg_benchmark.utils.miscellaneous import get_path

# main
def main(args):
    config_path = args.config
    weights = args.weights
    tracking = args.tracking
    rel_conf = args.rel_conf
    box_conf = args.box_conf
    dcs = args.dcs
    save_path = args.save_path
    visu_type = args.visu_type

    # this will create and load the model according to the config file
    # please make sure that the path in MODEL.WEIGHT in the config file is correct
    model = SGG_Model(config_path, weights, dcs=dcs, tracking=tracking, rel_conf=rel_conf, box_conf=box_conf)

    # Open the webcam
    cap = cv2.VideoCapture(-1)

    if save_path is not None:
        save_path = os.path.join(get_path(), save_path)
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        video_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 30, video_size)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Make prediction
        img, graph = model.predict(frame, visu_type=visu_type)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if visu_type == 'image':
            graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB)
            cv2.imshow('Graph', graph)
        #graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB)

        # Display the resulting frames
        cv2.imshow('Bbox detection', img)
        #cv2.imshow('Graph', graph)

        if save_path is not None:
            video_out.write(img)

        # if key "p" is pressed, pause the video
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.waitKey(-1)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if save_path is not None:
        video_out.release()

    # When everything is done, release the capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam demo")

    parser.add_argument('--config', default="configs/VG150/baseline/e2e_relation_X_101_32_8_FPN_1x.yaml", type=str, required=True, help='Path to the config file, e.g. config.yml')
    parser.add_argument('--weights', type=str, required=True, help='Path to the weights file, e.g. model.pth')
    parser.add_argument('--classes', default="datasets/vg/VG-SGG-dicts-with-attri.json", type=str, help='Path to the classes dict file, e.g. VG-SGG-dicts.json')
    parser.add_argument('--tracking', action="store_true", help='Object tracking or not')
    parser.add_argument('--rel_conf', type=float, default=0.01, help='Relation confidence threshold')
    parser.add_argument('--box_conf', type=float, default=0.25, help='Box confidence threshold')

    parser.add_argument('--dcs', type=int, default=100, help='Dynamic Candidate Selection')

    parser.add_argument('--save_path', type=str, default=None, help='Path to save the video')

    parser.add_argument('--visu_type', type=str, default='video', help='Visualization type: video or image')

    args = parser.parse_args()

    # change all relative paths to absolute
    if not os.path.isabs(args.config):
        args.config = os.path.join(get_path(), args.config)
    if not os.path.isabs(args.weights):
        args.weights = os.path.join(get_path(), args.weights)
    if not os.path.isabs(args.classes):
        args.classes = os.path.join(get_path(), args.classes)

    main(args)


# Run the demo
'''
python demo/webcam_demo.py --config checkpoints/PSG/SGDET/M-PE-NET-yolov8m/config.yml --weights checkpoints/PSG/SGDET/M-PE-NET-yolov8m/best_model_epoch_9.pth --dcs 42 --tracking --save_path ./output.avi

python demo/webcam_demo.py --config checkpoints/PSG/SGDET/penet-faster_rcnn/config.yml --weights checkpoints/PSG/SGDET/penet-faster_rcnn/best_model_epoch_1.pth --dcs 52 --tracking

'''