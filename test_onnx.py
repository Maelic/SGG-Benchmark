import cv2
import numpy as np
import onnxruntime as rt

# Load the ONNX model
sess = rt.InferenceSession("my_model.onnx")

# Start the webcam
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the image
    img = cv2.resize(frame, (640, 640))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)


    # Make a prediction
    input_name = sess.get_inputs()[0].name
    output_names = ['boxes', 'rels']
    pred_onx = sess.run(output_names, {input_name: img})
    print(pred_onx)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()