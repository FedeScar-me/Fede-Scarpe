# Hello human movement lovers!
Here you can find two .py files for "Indentifying Groups in Stroke Survivors", dataset here: https://www.kaggle.com/datasets/derekdb/toronto-robot-stroke-posture-dataset?resource=download.
1) main-file.py : run your code from this script.
Make sure to change the BASE_DATA_FOLDER variable to where you store the dataset.
"main-file.py" preprocesses the data by smoothing the trajectory on a frequence ratio, resamplling and interpolating eventual missing values.
It saves .npy(s) for movement x participant category. It calls the analysis function from "project1.py" (our second file) for both cathegories.
2) project1.py: runs the analysis.
It computes independent multidimensional DTW for each movement across all joints. It displayes distance scores (Silhouette, Average Intra-Cluster) for k = 2 -> n.
You can manipulate "chosen_k" to see how visualizations change: plot single joint 3D trajectory (with pca eigenvectors), plot per cluster average trajectories, plot per cluster limbs 2D trajectories.

# Enjoy :)
