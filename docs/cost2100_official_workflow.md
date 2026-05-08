# Official COST2100 Workflow for Exercise 2.15

This workflow is the canonical way to generate data for this project. The
assignment results in this repository are based on the official MATLAB
COST2100 model and the exported files under `data/cost2100_official`.

## 1. Clone COST2100

Clone the official repository outside or beside this project:

```powershell
git clone https://github.com/cost2100/cost2100.git "D:\NYCU\class\Artificial Intelligence Wireless\cost2100"
```

## 2. Test COST2100 in MATLAB

Open MATLAB and run:

```matlab
cd('D:\NYCU\class\Artificial Intelligence Wireless\cost2100\matlab')
addpath(genpath(pwd))
demo_model
```

If the demo runs, the COST2100 model is available.

## 3. Generate CsiNet-Compatible .mat Files

From this project root in MATLAB:

```matlab
cd('D:\NYCU\class\Artificial Intelligence Wireless\NYCU-AI-Wireless-Communication-HW\Exercise_2_15_ MidTerm')
generate_cost2100_csinet_datasets('D:\NYCU\class\Artificial Intelligence Wireless\cost2100')
```

The script exports:

```text
data/cost2100_official/D1_indoor_uniform/DATA_Htrain.mat
data/cost2100_official/D1_indoor_uniform/DATA_Hval.mat
data/cost2100_official/D1_indoor_uniform/DATA_Htest.mat
data/cost2100_official/D1_indoor_uniform/DATA_HtestF_all.mat
...
data/cost2100_official/D6_outdoor_clustered/...
```

The exported MATLAB keys are:

```text
HT     : normalized CsiNet input, shape [samples, 2048]
HF_all : complex frequency-domain CSI, shape [samples, 32, 125]
```

## 4. Validate Exported Files

In PowerShell:

```powershell
conda run -n csinet_tf python scripts/validate_cost2100_export.py --data-dir data/cost2100_official
```

## 5. Train and Evaluate CsiNet on Official COST2100 Data

```powershell
conda run -n csinet_tf python scripts/run_exercise_2_15_tf.py --data-dir data/cost2100_official --encoded-dim 512 --epochs 100 --batch-size 100 --mix-limit 1200 --val-limit 300
```

Then regenerate figures:

```powershell
conda run -n csinet_tf python scripts/plot_exercise_2_15_results.py --data-dir data/cost2100_official
```

## Important Adapter Note

The official COST2100 MATLAB API can differ by revision. If
`generate_cost2100_csinet_datasets.m` fails while extracting the channel
matrix, inspect the return value of:

```matlab
para = get_para('IndoorHall_5GHz');
channel = cost2100(para);
```

Then update only this function in `matlab/generate_cost2100_csinet_datasets.m`:

```matlab
extract_channel_matrix(channel)
```

The rest of the CsiNet pipeline can stay unchanged as long as the exported
files keep the same `HT` and `HF_all` layout.
