# Human activity HMM formative

Implementation prepared for the individual formative due 1 November 2026 at 23:59 CAT. **No personal sensor recordings or activity videos have been supplied. No real activity accuracy, report, or submission is claimed.**

## Run after collecting data

Use Python 3.9+ with NumPy and Matplotlib. The HMM algorithms use NumPy, with no HMM or scikit-learn implementation. Open `Activity_HMM.ipynb` for the reasoning and executable checks.

Keep whole recordings in `recordings/train/<activity>/*.csv` and independently collected unseen files in `recordings/test/<activity>/*.csv`. Activity folder names are `standing`, `walking`, `jumping`, and `still`. Do not split overlapping windows from one recording between training and testing. Prefer a later session for evaluation. Reserve at least two unseen files; represent every activity in the unseen set if all four sensitivity estimates are required.

The canonical CSV header is `timestamp,ax,ay,az,gx,gy,gz`. Timestamps are numeric seconds by default; use `--time-unit ms`, `us`, or `ns` when appropriate. Accelerometer and gyroscope channels must already be synchronized and use consistent physical units across files. A logger's original export should be preserved; inspect its actual headers and sensor timing before converting it to this schema. The parser rejects missing values, repeated/decreasing timestamps, large gaps, and files outside the required 5–10 second duration.

From this directory:

```sh
python3 -m unittest discover -s tests -v
python3 run_activity.py recordings results/session-01 --hz 20 --time-unit s
```

The result directory must be new. The run produces the fitted model and training scaler, recording quality information, per-window predictions, per-activity sensitivity/specificity/one-vs-rest accuracy, overall accuracy, confusion matrix, raw-signal examples, transition probabilities, emission density curves, EM convergence, and decoded test sequences. Undefined sensitivity for an absent activity is `null`, not an invented zero. The run refuses duplicate sensor content and stops if EM reaches its iteration limit without convergence.

## Design and reasoning

The pipeline harmonizes rates to an explicit common rate no higher than any source recording. It interpolates small timestamp jitter onto a regular source grid, applies a windowed-sinc low-pass filter before downsampling, and samples the common grid. Large jitter requires inspection rather than silently reconstructing missing motion. The default 20 Hz rate has a 10 Hz Nyquist limit; verify that this is suitable for the actual phones and activities. Raw rate, duration, and jitter are retained per file.

Two-second windows with a one-second hop provide 0.5 Hz FFT resolution while retaining several observations in a 5–10 second recording. Overlap improves temporal resolution but makes adjacent windows correlated. File separation protects evaluation; window counts are not independent participant counts. Incomplete tail windows are omitted.

Eleven features are extracted:

| Features | Why they may distinguish activity |
|---|---|
| Acceleration magnitude mean, standard deviation, RMS | Overall acceleration and movement variability; RMS retains signal strength |
| Gyroscope magnitude mean, standard deviation, RMS | Rotation and variability during movement |
| Accelerometer signal magnitude area | Combined axis motion; also sensitive to orientation |
| Acceleration and gyroscope dominant non-DC FFT frequency | Repetitive walking/jumping rhythm |
| Acceleration and gyroscope integrated spectral energy | Strength of time-varying motion after centering |

The FFT uses a Hann window and a one-sided PSD with correct interior-bin doubling. Spectral energy is the frequency integral of that PSD. Constant signals have zero non-DC peak frequency. Z-score parameters are fitted on training windows only and then reused unchanged on unseen data. Constant feature scales are replaced with one.

Four hidden states correspond to the named activities. Labelled training windows initialize diagonal Gaussian means and variances; small initial transition pseudocounts avoid impossible paths. Baum–Welch then performs log-domain forward/backward inference, sums sufficient statistics across independent files, updates parameters, and stops on a relative log-likelihood tolerance. A variance floor prevents singular Gaussian densities. An iteration cap is a guard, not evidence of convergence. State identities are aligned to training labels after fitting, without consulting test labels. Viterbi selects the most likely complete path; forward likelihood sums over every possible path.

The Gaussian emissions are densities, not probabilities of an exact continuous measurement. The emission figure shows marginal feature densities; the diagonal model assumes conditional feature independence, which is only an approximation because RMS, means, and energy can be correlated.

## Limits to discuss using real observations

Pure-activity files provide within-activity persistence but no observed transitions between different activities. File boundaries are never counted as activity transitions. Off-diagonal transitions therefore remain weakly identified. Do not describe the matrix as measured real-world switching behavior without collecting continuous sessions containing activity changes and their transition timestamps.

Standing while holding a phone and a phone resting on a table can be distinguished by grip noise or orientation rather than the intended human activity. The supplied brief uses these different placements. Record and report placement consistently, inspect this confound, and assess a separate session. Do not promise these two classes will be distinguishable before seeing data.

Synthetic tests check arithmetic and the full processing/output path. They do not estimate performance on personal recordings, other people, or different phone placements. The synthetic fixture exists only in temporary test directories and is never included as collected coursework data.

## Submission requirements still outstanding

- Collect the real labelled recordings. The brief mentions approximately 20 samples, while the excellent rubric asks for 50 files of 5–10 seconds per activity and at least 90 seconds per activity. Resolve that inconsistency with the facilitator before relying on the smaller count.
- Record four videos, one per activity, showing the entire body holding the recording phone for at least five seconds. The announcement says any missing activity video results in zero.
- Run the notebook and pipeline on the real data; inspect quality, convergence, confusion, and decoded paths. Do not tune using the final unseen set.
- Write the 4–5 page report with the required sections: background/use case; collection and preprocessing with video links; HMM setup/implementation; actual results/interpretation; discussion/conclusion. Personal reflection and real metrics cannot be supplied before the experiment.
- Put the labelled CSVs, notebook, implementation, and report in the required GitHub repository. Canvas accepts a `.doc` or `.docx` file. Code preparation alone is not this submission.

Source brief: https://alueducation.instructure.com/courses/3158/assignments/52339

Individual/video update: https://alueducation.instructure.com/courses/3158/discussion_topics/34642
