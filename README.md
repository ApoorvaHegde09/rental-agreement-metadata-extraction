# Rental Agreement Metadata Extraction

## About the Project

This project extracts important details from rental agreements and saves
them in a structured format.

It supports Word documents (`.docx`) and scanned images. The fields
extracted are:

-   Agreement Value
-   Start Date
-   End Date
-   Renewal Notice (Days)
-   Party One
-   Party Two

## How It Works

1.  Text is read from a Word document or extracted from an image using
    OCR.
2.  The text is split into tokens and converted into features.
3.  A Logistic Regression model identifies tokens that may belong to the
    required fields.
4.  Candidate dates and amounts are checked and the results are
    formatted as JSON or CSV.

The training labels are created by matching the values in the training
CSV files with the document text.

## Requirements

-   Python 3
-   Packages listed in `requirements.txt`
-   Tesseract OCR installed separately if you want to process scanned
    images

## Run the Project

Open a terminal in the `assignment-1` folder and install the required
packages:

``` bash
python -m pip install -r requirements.txt
```

Run extraction on the test documents:

``` bash
python scripts/predict.py --input data/test --truth data/test.csv --out outputs
```

Extract information from one document:

``` bash
python scripts/predict.py --input my_agreement.docx
```

To train the model again:

``` bash
python scripts/train.py --data data --out models
```

A trained model is already included in `models/extractor.joblib`.

## Optional REST API

Start the API:

``` bash
python scripts/serve.py --port 8000
```

Check that it is running at `http://localhost:8000/health`. The
extraction endpoint is `/extract`.

## Main Folders

-   `metaextract/` -- document reading, feature creation, model, and
    extraction logic
-   `scripts/` -- commands to train the model, run predictions, and
    start the API
-   `data/` -- training and test documents with CSV labels
-   `models/` -- saved trained model and evaluation files
-   `outputs/` -- prediction results and test metrics
