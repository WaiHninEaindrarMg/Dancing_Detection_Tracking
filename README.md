## Dancing-Detection-and-Tracking
Dancing detection and tracking involve the use of computer vision algorithms to identify and follow the movements of dancers within a video sequence.

The ability to track multiple dancers simultaneously makes dancing detection and tracking ideal for group performances, enabling synchronized movements and seamless coordination.

By leveraging machine learning models, dancing detection and tracking systems can accurately distinguish between dancers and background elements, ensuring precise tracking of human movements.

Real-time feedback provided by dancing detection and tracking systems enables performers to monitor their movements and improve their technique during rehearsals and live performances.

You can learn how to train about the custom multi-person detection model at my previous post (https://github.com/WaiHninEaindrarMg/Dancing_Detection.git). This project delves into the tracking aspect.


## Description
### Dancing-Detection-and-Tracking
Advanced machine learning models are trained on large datasets of annotated dance videos to improve the accuracy and robustness of dancing detection and tracking systems.

This project employs advanced image processing techniques to recognize and analyze the unique poses and gestures associated with different dance styles.

### Calculate Binary Mask Area (Remove Noise Detection)
For some of the false detections, We filtered out these false detections by calculating the binary mask area.

### Centroid Algorithm (Assign Track-IDs)
To achieve the Track-IDs, I utilized a centroid algorithm, which effectively handles occlusion scenarios often encountered during dance sequences. By employing this method, I was able to accurately track individuals throughout their movements, ensuring a comprehensive analysis of the dance sequences.

In instances where missed detections occur, I assign re-IDs to the track IDs. This process helps maintain continuity and accuracy in tracking individuals, even in cases where detections may have been temporarily lost or misidentified. By implementing re-IDs, I ensure a seamless tracking experience and improve the overall reliability of my system.


## Table of Contents
- [Installation](#installation)
- [Author](#author)
- [License](#license)


## Installation
1. Clone the repository:
```
git clone https://github.com/WaiHninEaindrarMg/Dancing_Detection_Tracking.git
```

2. Install Ultralytics , check here for more information (https://docs.ultralytics.com/tasks/segment/) :
```
pip install ultralytics
```

3. Custom Detection Model
I used my previous custom detection model from (https://github.com/WaiHninEaindrarMg/Dancing_Detection_Tracking/detection_model/best.pt)


## Instruction
1. Run this file https://github.com/WaiHninEaindrarMg/Dancing_Detection_Tracking/custom_tracking.py
```
python custom_tracking.py
```

After run this custom_tracking.py, video output will be showed.
This is result video (multi-person dancing detection and tracking results when occlusion conditions)
![Result](https://github.com/WaiHninEaindrarMg/Dancing_Detection_Tracking/blob/main/output/output_video.gif)

##
## Author
👤 : Wai Hnin Eaindrar Mg  
📧 : [waihnineaindrarmg@gmail.com](mailto:waihnineaindrarmg@gmail.com)



## License

This project is licensed under the [MIT License](LICENSE.md) - see the LICENSE file for details.

