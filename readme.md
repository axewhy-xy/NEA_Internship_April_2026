### Project Title: Automated Feature Extraction and Quantification System in HDB Residential Buildings for Wolbachia Technology Optimisation
Project Description: This study aims to identify HDB structures and estimate their impact on the wild-type mosquito population size and the efficacy of Wolbachia-Aedes releases.

### Project Outcome:
The outcome of the project would be a labelled dataset with structural and architectural features relevant to mosquito activity. The dataset is to be extracted using image processing and computer vision, validated for accuracy and cleaned for fitting causal machine learning model.

## Planned Tasked/TODO:
1. [X] - Set up environment for Google Earth Engine API for satellite imagery
2. [X] - Write an algorithm/script using the GEE API to retrieve satellite imagery.
3. [X] - Set up the environment for the Google Maps Street View API for close-up imagery/street view images of HDB buildings.
4. [X] - Write an algorithm/script using the Google Maps Street View API to retrieve street views of HDB buildings.
5. [X] - Set up enviroment for Google Map Tile API to obtain a Satellite Tile of the chosen location
6. [X] - Write an alogrithm/script to extract Satellite Tile and retrieve the images of the chosen location.
7. [ ] - Research and consider image labelling tools to be used for the collected dataset to feed into the chosen causal machine learning model.
8. [ ] - Label HDB features from the collected dataset. 
9. [ ] - Research and consider which computer vision model is suitable for the project usage. 
10. [ ] - Research and download the weights used to train the selected causal machine learning model in order to utilise transfer learning methodology. 
11. [ ] - Train the selected model with the collected dataset and tune hyperparameters to optimise the prediction accuracy.

## Problems Encountered: (Title: (date) [Complexity] [Status])
1. Low satellite imagery resolution: (2026-04-13) [Moderate] [Resolved]
While using Google Earth Engine API (GEE), one main problem encountered was the resolution of the satellite image. The root cause is the bottlenecking of using a dataset from Sentinel-2 SR, which has a 10m resolution for RGB bands. While the output image captures enough context around the selected coordinates/postal code, it is rather difficult to get a clear image of the HDB roofings for dataset labelling.

2. Converting postal codes to latitude and longitude: (2026-04-14) [Low] [Resolved]
The extraction script currently only takes in latitude and longtitude to get the image of the specific location. However, there is a need to figure out a method in order to convert postal address codes into the coordinate system, as most of the given data are in the form of postal code addresses.

## Mitigation/Solution:
1. Low satellite imagery resolution: (2026-04-13)
There are 3 possible solutions/mitigations that can solve this issue. The first solution would be to use a better resolution dataset, specifically a dataset from NICFI, as it provides a 4.77 m resolution which could potentially solve the low satellite image resolution. The second solution could be falling back to the old-school method of screenshotting individual images using Google Earth, which would take a long period of time for collecting a vast amount of satellite imagery. However, it is a well-proven method for obtaining a clear, high-resolution image which would be ideal for labelling the collected satellite image of the HDB buildings. Alternatively, instead of using satellite imagery, using a satellite tile or a 3D photorealistic tile could potentially work, bringing the best of both worlds with few compromises to the resolution. One API that can be considered will be using the Map Tile API from the Google Maps Platform which supports both photorealistic 3D Tiles for next generation visualization use cases as well as provideing satellite tile which could potentially solve the problem.

2. Converting postal codes to latitude and longitude: (2026-04-14)
One possible mitigation to this issue would be to painstakingly convert each postal address into lat/lon coordinates. However, this process would take a long period of time and effort, which is not feasible. Another possible solution could be using the OneMap API to convert postal addresses to lat/lon coordinates. This is a more feasible solution, but it may add more computation time to the extraction process. 

## Problem Resolution:
1. Low satellite imagery resolution:
Retrieved images by using Google Map Tile API instead.

2. Converting postal codes to latitude and longitude:
Use the OneMap API to convert postal address codes into latitude and longitude. As the extraction of the dataset would only be done a couple of times, the computation time could be negligible. 