# Fingerprint Verification System

### Requirements
1. OpenCV: ```pip3 install opencv-python```
---
2. OpenCV Contrib Python for image thinning function:
```pip3 install opencv-contrib-python```
3. Numpy
```
pip3 install numpy
```
4. Fingerprint Enhancer: Package used for fingerprint enhancement. It calculates frequency and orientation map for applying Gabor Filter on Normalized image. Since Frequency map calculation was not included in lecture modules and reference books, I used it for fingerprint enhancement purpose.
```
pip3 install fingerprint_enhancer
```



### Experiment 1
- Template Image: "FVC2000/DB2_B/101_1.tif"
- Query Image: "FVC2000/DB2_B/101_2.tif"
- Results
    - minutae count in template fingerprint: 109
    - minutae count in query fingerprint: 96
    - count of total matched minutae: 63


### Experiment 2
- Template Image: "FVC2000/DB2_B/101_1.tif"
- Query Image: "FVC2000/DB2_B/108_8.tif"
- Results:
    - minutae count in template fingerprint: 109
    - minutae count in query fingerprint: 315
    - count of total matched minutae: 0
