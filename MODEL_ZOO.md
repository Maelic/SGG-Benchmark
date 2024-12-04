# MODEL ZOO

## BACKBONES

| Dataset  | Faster-RCNN | mAP@50 (Faster-RCNN) | YOLOv8 | mAP@50 (YOLOv8) |
|----------|-------------|-------------------|--------|--------------|
| VG-150   | [Download](https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw) | 28.10 | [yolov8m_vg150.pt](https://drive.google.com/file/d/1cDxDU2fCs3eWqmmt1QbwZ7NxUnvWdoI7/view?usp=sharing) | 26.48 |
| IndoorVG | [Download](https://drive.google.com/file/d/1sY4dUAeAl18k6VZJgtuWhsnZNN0mW8PA/view?usp=sharing) | 25.31 | [yolov8m_indoorvg.pt](https://drive.google.com/file/d/15QBOVzXwsK0UX1DsMm2_IxWrdZf4nApl/view?usp=sharing) | 36.65 |
| PSG      | [Download](https://drive.google.com/file/d/1AL1fLMZsi_Q2MpsBtDxjOyE7zES6AMie/view?usp=sharing) | 35.38 | [yolov8m_psg.pt](https://drive.google.com/file/d/18xn56bSBAUiAxNhZ76U2tR5cjRnkW0oF/view?usp=sharing) | 53.60 |
| | | | [yolov8x_psg.pt](https://drive.google.com/file/d/1cZwQIzBOvaEPUSHXQ3UioTa18vti58dM/view?usp=sharing) | 57.20 |

## SGG Models

All models with YoloV8 and YoloV8-World are trained without any debiasing or re-weighting methods (such as TDE or reweight loss) and performance could probably be further improved. For instance

### VG150

New weights for the REACT model with YOLOV8-m (SGDET only):

Models | WEIGHTS | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | mAP | Latency (ms)
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
REACT (YOLOV8m)  | [Download](https://drive.google.com/file/d/1q7WAcJ9XS5ilt3Cf3ysBjwcz5CcaUzdJ/view?usp=sharing) | 21.04 | 26.16 | 28.75 | 9.78 | 12.26 | 13.63 | 31.8 | 23.9

Please download the weights in a ```checkpoints``` folder at the root of the codebase and run visualization using the ```demo/SGDET_on_cutom_images.ipynb``` notebook or evaluation using ```tools/relation_test_net.py```.

<!-- 

### INDOORVG

New weights for the REACT model with YOLOV8-m (SGDET only):

Models | WEIGHTS | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | 
-- | -- | -- | -- | -- | -- | -- | -- |


### PSG

New weights for the REACT model with YOLOV8-m (SGDET only):

Models | WEIGHTS | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | 
-- | -- | -- | -- | -- | -- | -- | -- |

 -->