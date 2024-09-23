# MODEL ZOO

## BACKBONES

| Dataset  | Faster-RCNN | mAP@50 (Faster-RCNN) | YOLOv8 | mAP@50 (YOLOv8) |
|----------|-------------|-------------------|--------|--------------|
| VG-150   | [Download](https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw) | 28.10 | [yolov8m_vg150.pt](https://drive.google.com/file/d/1cDxDU2fCs3eWqmmt1QbwZ7NxUnvWdoI7/view?usp=sharing) | 26.48 |
| IndoorVG | [Download](https://drive.google.com/file/d/1sY4dUAeAl18k6VZJgtuWhsnZNN0mW8PA/view?usp=sharing) | 25.31 | [yolov8m_indoorvg.pt](https://drive.google.com/file/d/15QBOVzXwsK0UX1DsMm2_IxWrdZf4nApl/view?usp=sharing) | 36.65 |
| PSG      | [Download](https://drive.google.com/file/d/1AL1fLMZsi_Q2MpsBtDxjOyE7zES6AMie/view?usp=sharing) | 35.38 | [yolov8m_psg.pt](https://drive.google.com/file/d/18xn56bSBAUiAxNhZ76U2tR5cjRnkW0oF/view?usp=sharing) | 53.60 |
| | | | [yolov8x_psg.pt](https://drive.google.com/file/d/1cZwQIzBOvaEPUSHXQ3UioTa18vti58dM/view?usp=sharing) | 57.20 |

## SGG Models

Coming soon...


<!-- 

All models with YoloV8 and YoloV8-World are trained without any debiasing or re-weighting methods (such as TDE or reweight loss) and performance could probably be further improved.

### VG150

New weights by me with YOLOV8 backbone and PE-Net model (SGDET only):

Models | WEIGHTS | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | 
-- | -- | -- | -- | -- | -- | -- | -- |
PE-NET YOLOV8m  | [Download](https://drive.google.com/file/d/1B1b6X7u7FV_R0dDDDbEgrGHXvKFjndN5/view?usp=sharing) | 22.47 | 27.87 | 30.63 | 8.59 | 11.19 | 12.46 |
PE-NET YOLOV8-Worldx | [Download](https://drive.google.com/file/d/1hW-Jucw29ffcBJaXDoVyUGLPDv3OyPmA/view?usp=sharing) | 16.96 | 21.69 | 24.05 | 7.59 | 9.54 | 10.41 |

The model with YOLOV8m object detector has better performance on VG150 but a bit worse performance on out-of-distribution images than YOLOV8-Worldx in my experiments. This is because YOLOV8-Worldx is fine-tuned from a large pre-training dataset and uses the CLIP text encoder so it is more robust.


Original model weights provided by Kaihua (a bit outdated) for Neural-Motifs-TDE with Faster-RCNN as a backbone (ResNeXt-101):

Models | WEIGHTS | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | zR@20 | zR@50 | zR@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
MOTIFS-SGDet-TDE    | [Download](https://1drv.ms/u/s!AmRLLNf6bzcir9x7OYb6sKBlzoXuYA?e=s3Y602) | 11.92 | 16.56 | 20.15 | 6.58 | 8.94 | 10.99 | 1.54 | 2.33 | 3.03
MOTIFS-SGCls-TDE    | [Download](https://1drv.ms/u/s!AmRLLNf6bzcir9xyuLO_I8TSZ6kfyQ?e=Y5686s) | 20.47 | 26.31 | 28.79 | 9.80 | 13.21 | 15.06 | 1.91 | 2.95 | 4.10
MOTIFS-PredCls-TDE  | [Download](https://1drv.ms/u/s!AmRLLNf6bzcir9xx725wYjN7lytynA?e=0B65Ws) | 33.38 | 45.88 | 51.25 | 17.85 | 24.75 | 28.70 | 8.28 | 14.31 | 18.04


### INDOORVG

New weights by me with YOLOV8 backbone and PE-Net model (SGDET only):

Models | WEIGHTS | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | 
-- | -- | -- | -- | -- | -- | -- | -- |
PE-NET YOLOV8m  | [Download](https://drive.google.com/file/d/1Adwc2W750uOxFqpwdD5AVOuQEp8aJDwH/view?usp=sharing) | 18.13 | 23.57 | 26.33 | 13.52 | 17.00 | 19.45 |

### PSG

New weights by me with YOLOV8 backbone and PE-Net model (SGDET only):

Models | WEIGHTS | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | 
-- | -- | -- | -- | -- | -- | -- | -- |
PE-NET YOLOV8m  | [Download](https://drive.google.com/file/d/1JZ36ftugJP2_W_U64WqF5YgylBRHKgX6/view?usp=sharing) | 26.57 | 29.85 | 31.34 | 18.06 | 19.96 | 20.81 | -->