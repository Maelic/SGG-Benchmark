import cv2
import numpy as np
import onnxruntime as rt
import seaborn as sns

def post_processing(pred_onx):
    bboxes = pred_onx[0]
    rels = pred_onx[1]

    # Filter boxes
    mask = bboxes[:, 4] > boxes_conf
    bboxes = bboxes[mask]

    # Modify the subj, obj to match the new indexes
    # print(mask)
    # for r in rels:
    #     r[0] = np.where(mask == True)[0][int(r[0])]
    #     r[1] = np.where(mask == True)[0][int(r[1])]

    # remove rels with object that are not in bboxes
    boxes_indexes = np.where(mask == True)[0]
    mask = np.isin(rels[:, 0], boxes_indexes) & np.isin(rels[:, 1], boxes_indexes)
    rels = rels[mask]

    # Filter rels
    mask = rels[:, 3] > rels_conf
    rels = rels[mask]

    return bboxes, rels

def visualize(pred_onnx, img, obj_classes, rel_classes):
    img_height, img_width, _ = img.shape
    text_padding = 2
    color = (0, 255, 0)
    boxes = pred_onnx[0][:,:4]
    # rescale boxes from 640,640 to img_height, img_width
    gain = min(640 / img_height, 640 / img_width)  # gain  = old / new

    boxes[..., :4] /= gain


    labels = pred_onnx[0][:,5]
    # to int
    labels = labels.astype(int)

    rels = pred_onnx[1]

    for i in range(len(rels)):
        sub = int(rels[i][0])
        obj = int(rels[i][1])

        obj_boxes = boxes[sub]
        sub_boxes = boxes[obj]

        c_obj =   draw_bbox(img, obj_boxes, obj_classes[labels[sub]], obj_classes)
        c_sub =   draw_bbox(img, sub_boxes, obj_classes[labels[obj]], obj_classes)

        cv2.line(img, c_sub, c_obj, color, 2)
        
        r_label = rel_classes[int(rels[i][2])]
        font_scale = 0.5

        # get text size
        (text_width, text_height), baseline = cv2.getTextSize(r_label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 1)
        # Draw background
        rect_start = ((c_sub[0] + c_obj[0]) // 2-2, ((c_sub[1] + c_obj[1]) // 2) - text_height - 2 * text_padding)
        rect_end = ((c_sub[0] + c_obj[0]) // 2 + text_width + 2 * text_padding, (c_sub[1] + c_obj[1]) // 2)

        # draw a rectange
        cv2.rectangle(img, rect_start, rect_end, color, cv2.FILLED)
        
        # Draw the relation label
        cv2.putText(img, r_label, ((c_sub[0] + c_obj[0]) // 2, (c_sub[1] + c_obj[1]) // 2 - 5), cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, font_scale, (255, 255, 255), 1)

    return img

def draw_bbox(img, bbox, label, obj_classes):
    obj_class_colors = sns.color_palette('Paired', len(obj_classes))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_padding = 2

    # Convert bbox to integer
    bbox = [int(b) for b in bbox]

    color = obj_class_colors[obj_classes.index(label)]

    # Draw bounding box
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # Determine the text to be drawn
    text = f"{label}"

    # Calculate text size (width, height) and baseline
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate rectangle coordinates such that the rectangle is inside the box, top left
    rect_start = (bbox[0], bbox[1] - text_height - 2 * text_padding)
    rect_end = (bbox[0] + text_width + 2 * text_padding, bbox[1])
    # if negative, move the rectangle to the left
    if rect_start[0] < 0:
        rect_start = (0, rect_start[1])
        rect_end = (text_width + 2 * text_padding, rect_end[1])
    if rect_end[0] > img.shape[1]:
        rect_start = (img.shape[1] - text_width - 2 * text_padding, rect_start[1])
        rect_end = (img.shape[1], rect_end[1])
    if rect_start[1] < 0:
        rect_start = (rect_start[0], 0)
        rect_end = (rect_end[0], text_height + 2 * text_padding)
    if rect_end[1] > img.shape[0]:
        rect_start = (rect_start[0], img.shape[0] - text_height - 2 * text_padding)
        rect_end = (rect_end[0], img.shape[0])

    # Draw background rectangle
    cv2.rectangle(img, rect_start, rect_end, color, cv2.FILLED)

    # Draw text
    cv2.putText(img, text, (rect_start[0] + text_padding, rect_end[1] - text_padding), font, font_scale, (255, 255, 255), font_thickness)

    # Return coordinates of the center of the bbox
    return (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2

# Load the ONNX model
sess = rt.InferenceSession("my_model.onnx")

# get model metadata
metadata = sess.get_modelmeta()


obj_classes = metadata.custom_metadata_map["obj_classes"]
rel_classes = metadata.custom_metadata_map["rel_classes"]
obj_classes = obj_classes[1:-1]
rel_classes = rel_classes[1:-1]

# from str to list of str
obj_classes = obj_classes.split(",")
rel_classes = rel_classes.split(",")

# remove "" and []
obj_classes = [obj_class.replace('"', '')[1:] for obj_class in obj_classes]
rel_classes = [rel_class.replace('"', '')[1:] for rel_class in rel_classes]

boxes_conf = 0.2
rels_conf = 0.1

# Start the webcam
cap = cv2.VideoCapture(0)
from torchvision import transforms

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the image
    img = cv2.resize(frame, (640, 640))
    # img = np.transpose(img, (2, 0, 1))
    # img = np.expand_dims(img, axis=0)
    # img = img.astype(np.float32)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    # to numpy
    img = img.numpy()

    # Make a prediction
    input_name = sess.get_inputs()[0].name
    output_names = ['boxes', 'rels']
    pred_onx = sess.run(output_names, {input_name: img})

    pred_onx = post_processing(pred_onx)

    # Visualize the prediction
    frame = visualize(pred_onx, frame, obj_classes, rel_classes)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
