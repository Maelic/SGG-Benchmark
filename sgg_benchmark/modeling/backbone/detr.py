from transformers import  DetrImageProcessor, DetrForObjectDetection

class DETRBackbone():
    """
    This class is a wrapper for the DETR backbone model,
    with a forward that extract features from last layers
    """
    def __init__(self) -> None:
        self.model = DetrImageProcessor()

    def features_extract(self, x):
        """
        This method extracts features from the last layers of the backbone
        """
        return self.model(x)