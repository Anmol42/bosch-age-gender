## Inter IIT Tech Meet : BOSCH PS 

Bosch Age and Gender Detection.

## Requirements (Versions used)

Pytorch - version 1.11.0  
Tensorflow - version 2.8.0  
Opencv-python - version 4.5.1.48  
Opencv-contrib-python - version 4.5.1.48  
Numpy - version 1.22.3  
Python - version 3.8.8  
mtcnn - version 0.1.1


## Steps to be followed

### To simulate the environment  
```bash
pip install -r requirements.txt
```

1. In the main folder run the following command in the terminal.
```bash
python3 test_inter.py --file="<path to the video file>" --fr_rate=<some positive integer>
```
2. To adjust sample rate in the video adjust the fr_rate parameter  
	E.g. fr_rate = 5 means every 5th frame will be used to predict persons, (can be used when the frame rate of original video is very high, we recommend a frame sample rate of 5-10 frames per second).
	
3. The output is a csv file generated named 'Outputs.csv'