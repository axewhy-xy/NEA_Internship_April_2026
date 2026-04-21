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
7. [ ] - Define what is the scope of 'feature extraction' in the context of this project, and determine the methodology for achieving the project outcome. This would     include discussing whether transfer learning can be utilised, and if so, which pre-trained models would be suitable for the task at hand. If transfer learning is not permitted, then there would be a need to discuss the specific features that need to be extracted from the images, and the methods for extracting those features from scratch.
8. [ ] - Write an algorithm/script to augment the dataset using techniques such as rotation, flipping, scaling, etc. to increase the diversity of the dataset and improve the performance of the model.
9a. [ ] - (if transfer learning is not permitted) Write an algorithm/script to extract the relevant features from the retrieved images using image processing techniques such as edge detection, corner detection, etc.
9b. [ ] - (if transfer learning is permitted) Label the dataset with the relevant features for training a pre-trained model.


## Problems Encountered: (Title: (date) [Complexity] [Status])
1. Low satellite imagery resolution: (2026-04-13) [Moderate] [Resolved]
While using Google Earth Engine API (GEE), one main problem encountered was the resolution of the satellite image. The root cause is the bottlenecking of using a dataset from Sentinel-2 SR, which has a 10m resolution for RGB bands. While the output image captures enough context around the selected coordinates/postal code, it is rather difficult to get a clear image of the HDB roofings for dataset labelling.

2. Converting postal codes to latitude and longitude: (2026-04-14) [Low] [Resolved]
The extraction script currently only takes in latitude and longtitude to get the image of the specific location. However, there is a need to figure out a method in order to convert postal address codes into the coordinate system, as most of the given data are in the form of postal code addresses.

3. Unclear project objective: (2026-04-20) [High] [Unresolved]
The project's purpose is unclear; there is a need to clarify the definition of feature extraction in the scope. Does 'feature extraction' mean HDB architectural features or does it mean the technical term 'feature extraction' in image processing? If it's the latter, then there will be a need to consult the methodology of achieving the project outcome. For instance, is it possible to utilise the concept of transfer learning using pre-existing models like ResNet-50 to aid with the feature extraction? If utilising transfer learning is permitted, then there is no need to work on extracting features like edge detection, corner detection, etc. from scratch. Instead, the task would then be updated to just dataset augmentation and labelling of the dataset to pass into the pre-trained model.

## Mitigation/Solution:
1. Low satellite imagery resolution: (2026-04-13)
There are 3 possible solutions/mitigations that can solve this issue. The first solution would be to use a better resolution dataset, specifically a dataset from NICFI, as it provides a 4.77 m resolution which could potentially solve the low satellite image resolution. The second solution could be falling back to the old-school method of screenshotting individual images using Google Earth, which would take a long period of time for collecting a vast amount of satellite imagery. However, it is a well-proven method for obtaining a clear, high-resolution image which would be ideal for labelling the collected satellite image of the HDB buildings. Alternatively, instead of using satellite imagery, using a satellite tile or a 3D photorealistic tile could potentially work, bringing the best of both worlds with few compromises to the resolution. One API that can be considered will be using the Map Tile API from the Google Maps Platform which supports both photorealistic 3D Tiles for next generation visualization use cases as well as provideing satellite tile which could potentially solve the problem.

2. Converting postal codes to latitude and longitude: (2026-04-14)
One possible mitigation to this issue would be to painstakingly convert each postal address into lat/lon coordinates. However, this process would take a long period of time and effort, which is not feasible. Another possible solution could be using the OneMap API to convert postal addresses to lat/lon coordinates. This is a more feasible solution, but it may add more computation time to the extraction process. 

3. Unclear project objective: (2026-04-15)
Consulting the project supervisor is the only solution to this issue. There is a need to discuss the project scope in detail in order to continue with the development progress. The scope of the discussion would be to clarify the definition of feature extraction in the context of this project, and to determine the methodology for achieving the project outcome. This would include discussing whether transfer learning can be utilised, and if so, which pre-trained models would be suitable for the task at hand. If transfer learning is not permitted, then there would be a need to discuss the specific features that need to be extracted from the images, and the methods for extracting those features from scratch. This discussion would be crucial in determining the direction of the project and ensuring that the development progress is aligned with the project objectives.

## Problem Resolution:
1. Low satellite imagery resolution:
Retrieved images by using Google Map Tile API instead. The image was retrieved using the satellite tile by calling the Google Map Tile API, which provided a much clearer image of the HDB roofings, allowing for better labelling of the dataset.

2. Converting postal codes to latitude and longitude:
Use the OneMap API to convert postal address codes into latitude and longitude coordinates. This allowed for a more efficient and automated way to retrieve the necessary coordinates for image extraction, streamlining the process and reducing the time and effort required for manual conversion. However, it is important to note that this solution would add more computation time to the extraction process, as each postal code would need to be converted before retrieving the corresponding image. This trade-off between automation and computation time would need to be considered when implementing this solution, especially if there is a large number of postal codes to process.

3. Unclear project objective: