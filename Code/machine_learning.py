## --------------------
##       Imports
## --------------------


## Library imports
from pathlib import Path
import cv2 as cv
import pandas as pd

## Machine learning related imports
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

## Existing functions imported
from Preprocessing import preprocess_retina
from feature_extraction import FEATURE_COLUMNS, extract_features


## --------------------
##  Modifiable Values
## --------------------


## Set of positive values for training
CANDIDATE_C_VALUES = (0.01, 0.1, 1, 10, 100)

## Path strings
FEATURES_PATH = "Features/features.csv"
SPLITS_PATH = "Features/splits.csv"
RESULTS_FOLDER = "SVM Results"


## --------------------
##   Preparing Data
## --------------------


## Create table that joins features and splits
def loadLabelledFeatures():
    ## Read CSVs
    features = pd.read_csv(FEATURES_PATH, dtype = {"image_id": str})
    splits = pd.read_csv(SPLITS_PATH, dtype = {"image_id": str})

    ## Merge on image_id and diagnosis
    labelledFeatures = features.merge(splits, on = ["image_id", "diagnosis"], indicator = True, how = "outer", validate = "one_to_one")

    ## Check if merged correctly
    if not labelledFeatures["_merge"].eq("both").all():
        raise ValueError("Some image IDs or diagnoses differ.")

    print("Labelled Features Table Created.")

    return labelledFeatures


## --------------------
##    Model Training
## --------------------


## Trains models and selects best one
def trainSVM(labelledFeatures):
    ## Initialize values
    bestModel = None
    bestScore = -1

    ## Get training features and data
    trainingData = labelledFeatures.loc[labelledFeatures["split"] == "train"] 
    trainingFeatures = trainingData[FEATURE_COLUMNS]
    trainingGrades = trainingData["diagnosis"]

    ## Get validation features and data
    validationData = labelledFeatures.loc[labelledFeatures["split"] == "val"] 
    validationFeatures = validationData[FEATURE_COLUMNS]
    validationGrades = validationData["diagnosis"]

    ## Test different models
    for cValue in CANDIDATE_C_VALUES:
        ## Create candidate model with current c value and fit with data
        candidateModel = make_pipeline(StandardScaler(), SVC(kernel = "rbf", C = cValue, class_weight = "balanced"))
        candidateModel.fit(trainingFeatures, trainingGrades)
        
        ## Predict validation grades and score
        predictedGrades = candidateModel.predict(validationFeatures)
        validationScore = f1_score(validationGrades, predictedGrades, labels = range(5), average = "macro", zero_division = 0)

        ## Update values when better model is found
        if validationScore > bestScore:
            bestScore = validationScore
            bestModel = candidateModel

    ## Print selected model
    print(f"Selected C = {bestModel.named_steps['svc'].C}, Validation Macro F1 = {bestScore}")

    return bestModel


## --------------------
##    Model Testing
## --------------------


## Create table for testing features for prediction
def CreateTestingFeatureTable():
    ## Initialize testing features list
    featuresData = []

    ## Go through each image in folder
    for imagePath in sorted(Path("Testing").iterdir()):

        ## Only process png files
        if imagePath.is_file() and imagePath.suffix.lower() == ".png":
            ## Extract testing features and add to list
            preprocessedImage = preprocess_retina(str(imagePath))
            imageFeatures = extract_features(preprocessedImage)

            featuresData.append({"image_id": imagePath.stem, **imageFeatures})

    ## Convert to table
    testingFeatures = pd.DataFrame(featuresData)
    return testingFeatures

## Predict grade results for testing images
def predictTestImages(model):
    ## Create testing features table and testing predictions
    testingFeatures = CreateTestingFeatureTable()
    testingPredictions = testingFeatures[["image_id"]].copy()

    ## Predict and set new values from testing and export CSV result
    testingPredictions["diagnosis"] = model.predict(testingFeatures[FEATURE_COLUMNS])
    testingPredictions["referral_flag"] = (testingPredictions["diagnosis"] >= 2).astype(int)

    testingPredictions.to_csv(f"{RESULTS_FOLDER}/testing_predictions.csv", index = False)

    ## Print path
    print(f"Predictions saved to {RESULTS_FOLDER}/testing_predictions.csv.")

    return None


## --------------------
##    Main Function
## --------------------


## Create main function
def main():
    ## Create results folder
    Path(RESULTS_FOLDER).mkdir(exist_ok = True)

    ## Label features and train model
    print("Creating Labelled Feature Table...")
    labelledFeatures = loadLabelledFeatures()
    
    print("Selecting Model...")
    model = trainSVM(labelledFeatures)

    ## Predict for test images
    print("Getting Test Predictions...")
    predictTestImages(model)

    return None

## Run main when file executed
if __name__ == "__main__":
    main()