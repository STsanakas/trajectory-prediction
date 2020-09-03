# trajectory-prediction

Code based on the [code by harvitronix](https://github.com/harvitronix/neural-network-genetic-algorithm)

Instructions:
1. Install [GeoPy Package](https://geopy.readthedocs.io/)
```
pip install geopy
```
2. Open TrajectoryPredictionLSTM.py and enter the path to the project folder.
The code assumes the following structure:

```bash

Project_Folder
├── Conv2Ang.py
├── Datasets
│   └── Demo.csv
├── Models
└── TrajectoryPredictionLSTM.py
```

eg. if you copy the project files in /home/user/TrajectoryPrediction, the path variable should be '/home/user/TrajectoryPrediction/'

3. Navigate to the Project Folder and run TrajectoryPredictionLSTM.py
```
python3 ./TrajectoryPredictionLSTM.py
```

You can change the dataset used in the second line of TrajectoryPredictionLSTM.py
