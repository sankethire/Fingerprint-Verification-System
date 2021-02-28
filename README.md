# Fingerprint Verification System

Original Fingerprint             |  Segmented Boundary    |   Segmented Image
:-------------------------:|:-------------------------:|:-------------------------:
![original fingerprint 1](https://user-images.githubusercontent.com/26035692/109406887-a1bb4680-79a2-11eb-9e23-f4338314cd05.png)  |  ![segmented boundary 1](https://user-images.githubusercontent.com/26035692/109406900-bd265180-79a2-11eb-8fa9-2e3637cf0617.png)  |  ![segmented image 1](https://user-images.githubusercontent.com/26035692/109406937-ffe82980-79a2-11eb-80f3-300be48e5038.png)

Normalized Image             |  Orientation Field    |   Enhanced Image
:-------------------------:|:-------------------------:|:-------------------------:
![normalized image 1](https://user-images.githubusercontent.com/26035692/109406955-2c9c4100-79a3-11eb-90e5-e7682ae14ea5.png)  |  ![orientation filed on normalized image 1](https://user-images.githubusercontent.com/26035692/109406965-43429800-79a3-11eb-9834-785b6d076b6a.png)  |  ![enhanced image 1](https://user-images.githubusercontent.com/26035692/109406981-640aed80-79a3-11eb-8e34-7d65011bb482.png)

Thinned Image             |  Minutiae Points (with false positives)    |   Minutiae Points (without false positives)
:-------------------------:|:-------------------------:|:-------------------------:
![thinned image 1](https://user-images.githubusercontent.com/26035692/109407052-e1366280-79a3-11eb-959b-77df5b65fbaa.png)  |  ![minutae points with false positives 1](https://user-images.githubusercontent.com/26035692/109407056-e7c4da00-79a3-11eb-9621-7115222355ba.png)  |  ![minutae points after false positives removal](https://user-images.githubusercontent.com/26035692/109407058-ec898e00-79a3-11eb-87d8-f117ac3b4028.png)


---

## Requirements
1. OpenCV: ```pip3 install opencv-python```
2. OpenCV Contrib Python (for image thinning function): ```pip3 install opencv-contrib-python```
3. Numpy: ```pip3 install numpy```
4. Fingerprint Enhancer: ```pip3 install fingerprint_enhancer```
    - This package is used for fingerprint enhancement. It calculates frequency and orientation map for applying Gabor Filter on Normalized image. Since Frequency map calculation was not included in lecture modules and reference books, I used it for fingerprint enhancement purpose.
5. FVC2000 database
---

## How to run the program

- Run the program using command: ```python3 main.py```
- In order to see the next image, use ```ESC``` key only, over the OpenCV Image window.

---

## Functions Implemented
(All Functions are implemented in "main.py" file.)
- Segmentation
- Normalization
- Orientation Map Calculation
- Minuatae Extraction Algorithm
- Alignment Algorithm
- Minutae Pairing Algorithm

---
## Outputs generated for minutae extraction
- Outputs are stored in "output images" folder
- It contains output folder for 3 fingerprint images from DB2_B (101_1.tif, 101_2.tif, 108_8.tif). sub folders are as following
    - DB2_B 101_1
    - DB2_B 101_2
    - DB2_B 108_8
- Each subfolders contain following tansformations of fingerprint images:
    - original fingerprint
    - segmented boundary
    - segmented image
    - normalized image
    - orientation map
    - enhanced image
    - thinned image
    - minutae points image with false positives
    - minutae points image after false positives removal

---

## Fingerprint Verification Experiments

### Experiment 1 (Different Fingerprints of Same Person)
- Template Image (Clear Fingerprint): "FVC2000/DB2_B/101_1.tif"
- Query Image (Clear Fingerprint): "FVC2000/DB2_B/101_2.tif"

- Experiment output images are located in:
    - "output images/DB2_B 101_1"
    - "output images/DB2_B 101_2"

- Results
    - minutae count in template fingerprint: 48
    - minutae count in query fingerprint: 36
    - count of total matched minutae: 30
    - match score:  0.7142857142857143
    (match score = 2*match_count/(template_minutae_count + query_minutae_count))

### Experiment 2 (Different Fingerprints of Different Persons)
- Template Image (Clear Fingerprint): "FVC2000/DB2_B/101_1.tif"
- Query Image (Noisy Fingerprint): "FVC2000/DB2_B/108_8.tif"

- Experiment output images are located in:
    - "output images/DB2_B 101_1"
    - "output images/DB2_B 108_8"

- Results:
    - minutae count in template fingerprint: 48
    - minutae count in query fingerprint: 166
    - count of total matched minutae: 47
    - match score:  0.4392523364485981
    (match score = 2*match_count/(template_minutae_count + query_minutae_count))
    

### Experiment 3 (Different Fingerprints of Different Persons)
- Template Image (Clear Fingerprint): "FVC2000/DB2_B/101_1.tif"
- Query Image(Clear Fingerprint): "FVC2000/DB2_B/102_3.tif"
- Results:
    - minutae count in template fingerprint: 48
    - minutae count in query fingerprint: 86
    - count of total matched minutae: 39
    - match score:  0.582089552238806
    (match score = 2*match_count/(template_minutae_count + query_minutae_count))

### Experiment 4 (Different Fingerprints of Different Persons)
- Template Image (Clear Fingerprint): "FVC2000/DB2_B/101_3.tif"
- Query Image(Clear Fingerprint): "FVC2000/DB2_B/105_6.tif"
- Results:
    - minutae count in template fingerprint: 49
    - minutae count in query fingerprint: 62
    - count of total matched minutae: 14
    - match score: 0.25225225225
    (match score = 2*match_count/(template_minutae_count + query_minutae_count))
---
