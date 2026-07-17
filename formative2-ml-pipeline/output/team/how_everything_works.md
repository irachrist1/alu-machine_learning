# How the Project Works

## 1. Tabular pipeline

`src/data_pipeline.py` loads the social-profile and transaction datasets. It removes exact duplicates, changes customer IDs such as `A178` to `178`, fills missing customer ratings with the median, and converts purchase dates to real date values.

Several social-profile rows can belong to one customer. The pipeline first summarizes them into one customer row. It then uses a validated many-to-one inner merge with transactions. This prevents duplicated transaction IDs. Date, engagement, interest, platform, and sentiment features are prepared for the product model.

## 2. Product recommendation model

`src/train_models.py` trains a Random Forest classifier to predict one of five product categories. Numeric values are median-imputed and categorical values are one-hot encoded. Customer rating is not used as an input because it is normally collected after a purchase.

The model is saved locally as `models/product_model.joblib`. Its held-out accuracy is 0.167, macro F1 is 0.177, and log loss is 1.746. These results are low because the dataset has weak relationships between the available inputs and product category.

## 3. Image pipeline and facial recognition

`src/image_pipeline.py` reads each member's images from their named folder. Every original image is converted to RGB and resized consistently. The script creates three augmentations: rotation, horizontal flip, and grayscale.

It extracts color and grayscale histograms, brightness statistics, and a small grayscale pixel embedding. The result is saved in `data/processed/image_features.csv`. `src/train_models.py` trains a Random Forest facial identity model from those features.

During authentication, the predicted face confidence must meet the default 85% threshold. A lower score immediately returns `Access Denied`.

## 4. Audio pipeline and voice verification

`src/audio_pipeline.py` converts recordings to mono 16 kHz audio. It creates three augmentations: pitch shift, time stretch, and low-level background noise.

The script extracts the mean and standard deviation of 13 MFCC coefficients, spectral roll-off, RMS energy, zero-crossing rate, and duration. The result is saved in `data/processed/audio_features.csv`. A Random Forest model learns each member's voiceprint.

## 5. Command-line system flow

`src/cli_app.py` runs the security checks in this order:

1. Predict the identity from the face image.
2. Deny access if face confidence is below the threshold.
3. Run the product model, but keep its prediction hidden.
4. Predict the identity from the voice sample.
5. Deny access if voice confidence is too low.
6. Compare the face identity with the voice identity.
7. Display the predicted product only when both identities match.

A non-face image tests the first denial route. A valid Christian face combined with Mahlet's audio tests the identity-mismatch route. Christian's valid face and voice test the approved route.

## 6. Notebook, tests, and report

The notebook shows cleaning, merge checks, at least three EDA plots, media samples, augmentations, model metrics, and the approved and denied demonstrations. The tests check the feature files, saved models, merge rules, and command-line outcomes. The report summarizes the same evidence in submission form.

## 7. What to rerun after team uploads

Place each member's three images and two audio recordings in the expected named folders. Then follow the commands in `README.md` to regenerate image features, regenerate audio features, retrain the models, run tests, and rebuild the executed notebook and report.

Do not add `reports/video_demo_script.md` or any file whose name starts with `video_demo_script` to Git. Those recording notes are local-only and are covered by `.gitignore`.
