# MODEL ZOO

## BACKBONES

| Dataset  | Faster-RCNN | YOLOv8 |
|----------|-------------|--------|
| VG-150   | [Download](https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw) | [yolov8m_vg150.pt](https://drive.google.com/file/d/1cDxDU2fCs3eWqmmt1QbwZ7NxUnvWdoI7/view?usp=sharing)
| IndoorVG | [Download](https://drive.google.com/file/d/1sY4dUAeAl18k6VZJgtuWhsnZNN0mW8PA/view?usp=sharing) | [yolov8m_indoorvg.pt](https://drive.google.com/file/d/15QBOVzXwsK0UX1DsMm2_IxWrdZf4nApl/view?usp=sharing)
| PSG      | [Download](https://drive.google.com/file/d/1AL1fLMZsi_Q2MpsBtDxjOyE7zES6AMie/view?usp=sharing) | [yolov8m_psg.pt](https://drive.google.com/file/d/18xn56bSBAUiAxNhZ76U2tR5cjRnkW0oF/view?usp=sharing)

## SGG Models

### VG150

Original model weights provided by Kaihua (a bit outdated) for Neural-Motifs-TDE:

| PREDCls | SGCls | SGDet |
|---------|-------|-------|
| [Download](https://1drv.ms/u/s!AmRLLNf6bzcir9xx725wYjN7lytynA?e=0B65Ws) | [Download](https://1drv.ms/u/s!AmRLLNf6bzcir9xyuLO_I8TSZ6kfyQ?e=Y5686s) | [Download](https://1drv.ms/u/s!AmRLLNf6bzcir9x7OYb6sKBlzoXuYA?e=s3Y602) |

Corresponding metrics:

Models |  R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | zR@20 | zR@50 | zR@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
MOTIFS-SGDet-TDE    | 11.92 | 16.56 | 20.15 | 6.58 | 8.94 | 10.99 | 1.54 | 2.33 | 3.03
MOTIFS-SGCls-TDE    | 20.47 | 26.31 | 28.79 | 9.80 | 13.21 | 15.06 | 1.91 | 2.95 | 4.10
MOTIFS-PredCls-TDE  | 33.38 | 45.88 | 51.25 | 17.85 | 24.75 | 28.70 | 8.28 | 14.31 | 18.04
