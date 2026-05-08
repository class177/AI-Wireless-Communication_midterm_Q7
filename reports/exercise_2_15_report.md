# Exercise 2.15: CsiNet on Other Channel Datasets

## Problem Translation

Exercise 2.15 asks how CsiNet performs on channel datasets other than the original training distribution. The tasks are:

1. Use the COST 2100 channel model to generate more than five different channel datasets, for example by changing the distribution of users.
2. Evaluate the CSI reconstruction NMSE of a trained CsiNet model on each dataset.
3. Mix the different channel datasets and use the mixed data to train CsiNet. Compare the reconstruction performance with the result in part (b), and discuss how to improve the generalization of CSI feedback methods in practical systems.

## Reference Setting

The CsiNet reference paper uses the COST 2100 channel model with the following important settings:

- Indoor picocellular scenario at 5.3 GHz.
- Outdoor rural scenario at 300 MHz.
- Base station at the center of a square region.
- User equipment randomly distributed in the square region.
- Indoor region side length: 20 m.
- Outdoor region side length: 400 m.
- Uniform linear array with 32 BS antennas.
- 1024 OFDM subcarriers.
- After 2D DFT, only the first 32 delay-domain rows are retained, giving a 32 x 32 angular-delay channel matrix.
- The CsiNet input contains two channels: real and imaginary parts, normalized to [0, 1].
- NMSE is defined as `E{||H - H_hat||^2 / ||H||^2}` and is reported in dB.

## Experiment Version Used in This Repository

This report is aligned with the current repository results in `README.md` and `result/exercise_2_15_csinet_results.csv`.

- Channel source: official COST2100 MATLAB model exported through `matlab/generate_cost2100_csinet_datasets.m`
- Data directory: `data/cost2100_official`
- Model: TensorFlow/Keras CsiNet
- Encoded dimension: `512`
- Training epochs: `100`
- Single-dataset training samples: `1200`
- Validation samples per dataset: `300`
- Test samples per dataset: `400`
- Mixed training samples: `7200` total, from all six datasets

## (a) Dataset Generation

Six different datasets were generated from the official COST2100 workflow. They preserve the CsiNet input format `32 x 32 x 2`, while changing the user distribution and propagation condition.

| Dataset | COST2100 Environment | User Distribution | Purpose |
|---|---|---|---|
| D1_indoor_uniform | `IndoorHall_5GHz` | Uniform indoor users | Baseline indoor distribution |
| D2_indoor_center | `IndoorHall_5GHz` | Users concentrated near the BS | Tests near-user indoor channels |
| D3_indoor_edge | `IndoorHall_5GHz` | Users near the cell edge | Tests far-user indoor channels |
| D4_indoor_ring | `IndoorHall_5GHz` | Ring-shaped indoor distribution | Tests structured indoor variation |
| D5_outdoor_uniform | `SemiUrban_300MHz` | Uniform outdoor users | Tests outdoor domain shift |
| D6_outdoor_clustered | `SemiUrban_300MHz` | Clustered outdoor users | Tests outdoor hotspot-like deployment |

The exported `.mat` files use:

- `HT`: normalized CsiNet input, shape `[samples, 2048]`
- `HF_all`: complex frequency-domain CSI for testing, shape `[samples, 32, 125]`

## (b) Cross-Dataset Evaluation

The baseline CsiNet model is trained on `D1_indoor_uniform` and then tested on all six datasets. The TensorFlow implementation uses `channels_last` internally for Windows CPU compatibility, while keeping the CSI tensor content and NMSE/rho evaluation equivalent to the CsiNet setting.

| Train Dataset | Test Dataset | NMSE (dB) | rho |
|---|---|---:|---:|
| D1_indoor_uniform | D1_indoor_uniform | -8.5760 | 0.139362 |
| D1_indoor_uniform | D2_indoor_center | -7.5798 | 0.138616 |
| D1_indoor_uniform | D3_indoor_edge | -8.6165 | 0.149060 |
| D1_indoor_uniform | D4_indoor_ring | -8.7081 | 0.143861 |
| D1_indoor_uniform | D5_outdoor_uniform | -3.4702 | 0.136041 |
| D1_indoor_uniform | D6_outdoor_clustered | -3.7739 | 0.144101 |

The baseline model performs reasonably on the indoor datasets, but its NMSE degrades strongly on the outdoor datasets. This shows that a model trained only on one indoor distribution does not generalize well when the channel statistics shift to another environment.

## (c) Mixed-Dataset Training and Comparison

The six datasets were mixed and used as the training set. The mixed model was then evaluated on each individual test dataset.

| Test Dataset | Baseline NMSE (dB) | Mixed-Train NMSE (dB) | Improvement (dB) | Baseline rho | Mixed rho |
|---|---:|---:|---:|---:|---:|
| D1_indoor_uniform | -8.5760 | -12.1743 | 3.5983 | 0.139362 | 0.143565 |
| D2_indoor_center | -7.5798 | -11.1076 | 3.5278 | 0.138616 | 0.141723 |
| D3_indoor_edge | -8.6165 | -12.4335 | 3.8170 | 0.149060 | 0.150543 |
| D4_indoor_ring | -8.7081 | -12.1716 | 3.4635 | 0.143861 | 0.145666 |
| D5_outdoor_uniform | -3.4702 | -12.9349 | 9.4647 | 0.136041 | 0.148281 |
| D6_outdoor_clustered | -3.7739 | -13.1864 | 9.4125 | 0.144101 | 0.155575 |

The mixed-training model improves NMSE on every test dataset. The gain is about `3.46-3.82 dB` on indoor datasets and about `9.41-9.46 dB` on outdoor datasets. This supports the conclusion that training only on one channel distribution limits generalization, while training on diverse channel realizations makes the CSI feedback model much more robust under domain shift.

## Discussion

In practical wireless systems, the channel distribution changes with user location, carrier frequency, scattering environment, mobility, and deployment geometry. A CsiNet model trained on a single scenario may reconstruct CSI well only for channels similar to the training data. When the test distribution shifts, the learned encoder and decoder may no longer preserve the most important angular-delay components, causing NMSE degradation.

To improve generalization in practical systems, the CSI feedback model should be trained with diverse channel data. Useful strategies include:

- Mix indoor, outdoor, cell-center, cell-edge, and clustered channel samples during training.
- Use training datasets that span multiple propagation environments and user distributions.
- Fine-tune the model when deployment-specific CSI data becomes available.
- Use transfer learning or domain adaptation when moving to a new scenario with limited labeled data.
- Extend the model to exploit temporal correlation when users are mobile.

## Reproduction Commands

Generate the official COST2100 data in MATLAB:

```matlab
cd('D:\NYCU\class\Artificial Intelligence Wireless\NYCU-AI-Wireless-Communication-HW\MidTerm_Q7')
addpath(genpath(fullfile(pwd, 'matlab')))
cost_root = fullfile(pwd, 'cost2100');
addpath(genpath(fullfile(cost_root, 'matlab')))
generate_cost2100_csinet_datasets(cost_root)
```

Validate the exported data:

```powershell
conda run -n csinet_tf python scripts/validate_cost2100_export.py --data-dir data/cost2100_official
```

Run the TensorFlow CsiNet experiment:

```powershell
conda run -n csinet_tf python scripts/run_exercise_2_15_tf.py --data-dir data/cost2100_official --encoded-dim 512 --epochs 100 --batch-size 100 --mix-limit 1200 --val-limit 300
```

## Output Files

The main output files are:

- `result/exercise_2_15_csinet_results.csv`
- `result/history_CsiNet_D1_indoor_uniform_dim512_epochs100.csv`
- `result/history_CsiNet_mixed_all_dim512_epochs100.csv`
- `saved_model/CsiNet_D1_indoor_uniform_dim512.weights.h5`
- `saved_model/CsiNet_mixed_all_dim512.weights.h5`
