# Fingerprint Verification System

## Requirements
1. OpenCV: ```pip3 install opencv-python```
2. OpenCV (Contrib Python for image thinning function): ```pip3 install opencv-contrib-python```
3. Numpy: ```pip3 install numpy```
4. Fingerprint Enhancer: ```pip3 install fingerprint_enhancer```
    - This package used for fingerprint enhancement. It calculates frequency and orientation map for applying Gabor Filter on Normalized image. Since Frequency map calculation was not included in lecture modules and reference books, I used it for fingerprint enhancement purpose.
---

## Fingerprint Verification Experiments

### Experiment 1 (Different Fingerprints of Same Person)
- Template Image (Clear Fingerprint): "FVC2000/DB2_B/101_1.tif"
- Query Image (Clear Fingerprint): "FVC2000/DB2_B/101_2.tif"

- Results
    - minutae count in template fingerprint: 48
    - minutae count in query fingerprint: 36
    - count of total matched minutae: 30


### Experiment 2 (Different Fingerprints of Different Persons)
- Template Image (Clear Fingerprint): "FVC2000/DB2_B/101_1.tif"
- Query Image (Noisy Fingerprint): "FVC2000/DB2_B/108_8.tif"
- Results:
    - minutae count in template fingerprint: 48
    - minutae count in query fingerprint: 166
    - count of total matched minutae: 47

### Experiment 3 (Different Fingerprints of Different Persons)
- Template Image (Clear Fingerprint): "FVC2000/DB2_B/101_1.tif"
- Query Image(Clear Fingerprint): "FVC2000/DB2_B/102_3.tif"
- Results:
    - minutae count in template fingerprint: 48
    - minutae count in query fingerprint: 86
    - count of total matched minutae: 39

### Experiment 4 (Different Fingerprints of Different Persons)
- Template Image (Clear Fingerprint): "FVC2000/DB2_B/101_3.tif"
- Query Image(Clear Fingerprint): "FVC2000/DB2_B/105_6.tif"
- Results:
    - minutae count in template fingerprint: 49
    - minutae count in query fingerprint: 62
    - count of total matched minutae: 14


---
## How to run the program

```
python3 main.py
```