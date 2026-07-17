# Group 19 Media Checklist

Every group member must upload their own data to the shared Google Drive folder.

Shared folder: https://drive.google.com/drive/folders/1K61vL1wpahKuZmzpgbmyYaSKZufWVr4z

## Required files per member

### Images

Create a folder with your full name under `Images` and upload:

1. `neutral.jpg` - normal face with no smile.
2. `smiling.jpg` - clearly smiling.
3. `surprised.jpg` - eyes open and a clear surprised expression.

The face should be close to the camera, looking forward, and clearly visible. Use good lighting and do not use filters.

### Audio

Create a folder with the same full name under `Audios` and upload:

1. `yes_approve.m4a` - say exactly: “Yes, approve.”
2. `confirm_transaction.m4a` - say exactly: “Confirm transaction.”

Record each phrase separately in a quiet room. Speak clearly and keep a short silence before and after the phrase.

## Current status

| Member | Images | Audio | Remaining work |
|---|---:|---:|---|
| Christian Tonny Gentil Iradukunda | 3/3 | 2/2 | Complete |
| Hassan | 0/3 | 0/2 | Upload all five files |
| Mahlet Tilahun | 0/3 | 2/2 | Upload three facial expressions |
| Yvette Uwimpaye | 1/3 | 0/2 | Upload the three required close-up expressions and both audio phrases |

Yvette's existing image is useful as an extra sample, but it does not replace the required neutral, smiling, and surprised close-up photographs.

## What happens after uploading

The image script creates rotation, horizontal flip, and grayscale versions of every image, then saves histogram and pixel embedding features to `image_features.csv`.

The audio script creates pitch shift, time stretch, and background noise versions of every recording, then saves MFCC, spectral roll-off, energy, and other features to `audio_features.csv`.

The face and voice models are retrained after all files are present. The final command-line demonstration checks the face, runs the product model, checks the voice, and displays the prediction only when both identities match.
