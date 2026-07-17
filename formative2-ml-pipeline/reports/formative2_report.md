# Formative 2: Multimodal Data Preprocessing

**Group:** 19  
**Project:** User Identity and Product Recommendation System  
**Prepared by:** Christian Tonny Gentil Iradukunda and Group 19

## 1. Introduction

This project simulates a secure product recommendation system. A face image is checked first. If the face is accepted, the product model runs but does not immediately show its answer. A voice sample is then checked. The recommendation is only displayed when the face and voice identities agree. Failed checks return Access Denied.

The work combines two tabular datasets with face images and short audio recordings. It covers data cleaning, merge validation, exploratory analysis, feature engineering, augmentation, model training, evaluation, and a command-line demonstration.

## 2. Tabular data preprocessing

The customer social profile dataset originally had 155 rows and 5 exact duplicates. After removing the duplicates, it had 150 social observations belonging to 84 unique customers. Customer identifiers such as A178 were converted to the numeric form 178 so they could match the transaction data.

The transaction dataset had 150 rows. Ten customer rating values were missing, so they were filled using the median rating of 3.0. Purchase dates were converted to datetime values. Transaction IDs and product categories had no missing values.

Some customers had several social observations. A direct join would duplicate their transactions, so social data was first aggregated to one row per customer. The engineered customer features included mean and maximum engagement, mean and maximum purchase interest, social profile count, platform count, strongest platform, and modal review sentiment.

An inner many-to-one merge was used because the model needs data from both sources. The merge produced 117 transactions. Thirty-three transaction rows were excluded because their customers did not have a social profile. Post-merge checks confirmed that no transaction ID was duplicated.

Date features were added for purchase month, day of week, and quarter. Customer rating was kept in the cleaned dataset but excluded from the model because ratings are collected after a purchase and could cause leakage.

## 3. Exploratory data analysis

The merged target was fairly balanced: Sports had 28 rows, Electronics 27, Clothing 22, Groceries 20, and Books 20. Purchase amount ranged from 62 to 495, with a mean of 287.91. The mean engagement score was 74.15 and mean purchase interest was 3.08.

The distribution, boxplot, and correlation plots showed considerable overlap between product categories. Numeric correlations were weak. This suggested that the available features would not strongly predict the product category.

## 4. Image processing

Images were arranged in one folder per member. Every image was converted to RGB and fitted to a consistent size. Three augmented variants were created from each original: a 15-degree rotation, horizontal flip, and grayscale conversion.

The image feature file contains normalized red, green, blue, and grayscale histograms, mean and standard deviation of brightness, and a 16 by 16 grayscale pixel embedding. There are 322 numeric image features per row.

The available local data contained three Christian images and one Yvette image. The four originals produced 16 rows after adding the original and three variants. Augmented copies from one original were kept on the same side of the model split to reduce leakage.

## 5. Audio processing

Phone audio files were decoded to mono audio at 16 kHz. Three augmentations were produced for every original: a two-semitone pitch shift, a 0.90 time stretch, and low-level background noise.

Audio features included the mean and standard deviation of 13 MFCC coefficients, spectral roll-off, RMS energy, zero-crossing rate, and duration. This produced 30 numeric features per row.

Christian and Mahlet each had two original recordings. The four originals produced 16 feature rows after augmentation. Waveforms showed clear speech sections, while the spectrograms showed that most speech energy was concentrated in the lower frequencies.

## 6. Models and evaluation

Three Random Forest classifiers were implemented.

| Model | Accuracy | Macro F1 | Log loss |
|---|---:|---:|---:|
| Product recommendation | 0.167 | 0.177 | 1.746 |
| Facial recognition | 1.000 | 1.000 | 0.088 |
| Voiceprint verification | 1.000 | 1.000 | 0.096 |

The product model was evaluated on 30 held-out rows. Its low result agrees with the weak relationships seen in EDA. It is functional, but the result should not be treated as production quality.

The face and voice models used original-file grouping so an augmented copy of a test sample was not used during training. The voice evaluation included Christian and Mahlet. The image evaluation only included Christian because Yvette had one original image. For this reason, the perfect image score is incomplete and should not be generalized. The authentication datasets are also very small, which can make results unstable.

## 7. System demonstration

The command-line application follows the required order:

1. Read the face image and predict the identity.
2. Reject low-confidence face predictions.
3. Run the product recommendation model but hold the answer.
4. Read the voice sample and predict the speaker.
5. Display the product only if the face and voice identities match.

In the successful test, Christian's neutral face scored 89.7% confidence and his “Yes, approve” audio scored 99.3%. The system approved the transaction and predicted Clothing.

Two denial tests were also completed. A non-face image scored below the 85% face threshold and was rejected before product display. In another test, Christian's face was combined with Mahlet's voice. Both samples were individually recognized, but their identities did not match, so the system returned Access Denied.

## 8. Contributions

| Member | Contribution |
|---|---|
| Christian Tonny Gentil Iradukunda | Submitted three facial expressions and two voice phrases; completed data cleaning, merge logic, feature engineering, model integration, testing, CLI simulation, notebook, and report. |
| Hassan | Created and shared the group data folders and coordinated member uploads. |
| Mahlet Tilahun | Submitted two voice recordings used in audio preprocessing and voice model training. |
| Yvette Uwimpaye | Submitted one image used in image preprocessing and face model training. |

## 9. Limitations and conclusion

The system demonstrates a complete multimodal sequence and both approved and denied paths. The merge was validated, required feature CSV files were created, and all models report accuracy, macro F1-score, and log loss.

The largest limitation is the amount of media. At the time of processing, Hassan had not uploaded media, Mahlet had no images, and Yvette had one image and no audio. The product data also contains weak predictive signals. More original samples per member and more customer history would be required for a stronger real-world system.

## 10. Submission links

- GitHub repository: **https://github.com/irachrist1/alu-machine_learning**
- System demonstration video: **Add the uploaded video URL before submission.**

## Data sources

- Customer social profiles: https://docs.google.com/spreadsheets/d/10up-WdC0a6egYaXLKiMQUotpOvaKZRZvYuWFQx4-RPQ/
- Customer transactions: https://docs.google.com/spreadsheets/d/1s4WOVm49lmLQ8d9QbbbdgAcRTNh3m0KCH_5ciiaRZw0/
