# Detecting Vocal Fold Paralysis (VFP) with machine learning

**Cite this article if using the data:**

Low, D. M., Randolph, G., Rao, V., Ghosh, S. S. & Song, P., C. (2023). Uncovering the important acoustic features for detecting vocal fold paralysis with explainable machine learning. MedRxiv. 


### Tutorial to test our models

`vfp_detector.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielmlow/vfp/blob/main/vfp_detector.ipynb) 


# 1. Data

Note: 
* "speech" in this repo refers to "reading" task in the manuscript.
* The original audio wav files cannot be shared due to consent restrictions. Here we provide the extracted eGeMAPS features (see manuscript for details).


### 1.1. Demographic information 
* `./data/input/VFP_DeidentifiedDemographics.csv` De-identified demographic information
 
### 1.2. eGeMAPS features, ids, and labels 
**Columns:**
* `sid` subject ID. Important for group shuffle split.
* `filename` from which wav file were features extracted
* `token` what type of sample (speechN, vowelN where N is sample number)
* `target` label to be predicted

```
egemaps_vector_both.csv
egemaps_vector_speech.csv
egemaps_vector_vowel.csv
```




# 2. Code and instructions for reproducibility 
 
To run the `.py` (inluding pydra-ml package) or the `.ipynb` on Jupter Notebook, create a virtual environment and install the `requirements.txt`:
* `conda create --name pydra_vfp --file requirements.txt`
* `conda activate pydra_vfp`


### Figure 1:
* `./fig_1.ipynb/`
* `./data/input/rainbow.wav` audio file used
* `./data/input/rainbow_f0.txt` f0 over time from PRAAT 

### Table 1: Sample sizes and demographic information. 
* `./data/input/VFP_DeidentifiedDemographics.csv` De-identified demographic information
* `demographics.py` Script to obtain info for Table 1.

### Figure 2:
* `duration.ipynb` 

### Performance results: Figure 3 and Table 2, and Sup. Figure S10
We ran models using [pydra-ml](https://github.com/nipype/pydra-ml) for which a spec file is needed where the dataset is specified. The dataset needs to be in the same dir where the spec file is run. Since we ran models on a cluster, we have SLURM scripts, so  the dataset is in the same dir as the SLURM scripts.
`if` and `indfact` stand for independence factor, the algorithm we created for removing redundant features. 
* `./vfp_v7_indfact/` dir 
    * `/specs/` spec files
    * `run_collinearity_job_array_{data_type}_if.sh` SLURM script to run pydra-ml spec files where `data_type` is 'speech', 'vowel' or 'both'. 
        * ```$ pydraml -s specs/vfp_spec_4models_both_if_{spec_id}.json``` where `spec_id` is value in `range(1,10)` corresponding to dcorrs thresholds of `np.arange(0.2, 1.1, 0.1)` (i.e., we removed redudant features according to the dcor threshold). The job array runs the different spec files in parallel.
    * `thresholds_if_Nvars_{data_type}.txt` were used to build those spec files.
    * `run_clear_locks.sh` runs `clear_locks.py` Run this this if you want to re-run model with different specs (pydra-ml will re-use cache-wf)
    * `run_collinearity_speech_explanations.sh` re-runs models setting `gen_shap` to true in spec files to output SHAP values/explanations.
    * `./outputs/` each run will output a dir such as `out-vfp_spec_4models_both_if_1.json-20200910T024552.823868` with the name of the spec file. 
    
* `performance_stats.py` p-values in Figure 3
* `./vfp_v8_top5/` runs top 5 featurs specificed in spec files
* `analyze_results.py` takes `outputs/out-*` files from pydra-ml and produces summaries which were then concatenated into table 2. Also figures for Sup. Figure S10. 
* `cpp.ipynb` CPP models
* `duration.ipynb` Duration models


### Figure 4
* `shap_analysis.ipynb` Parallel coordinate plots using SHAP scores.

### Figure 5
* `./vfp_v8_top1outof5/` runs one of the top 5 features at a time.
* `shap_analysis.ipynb` makes the plots

### Figure 6 and Sup. Fig. S1-S9
* `collinearity.py` remove redudant features (reduce multicollinearity) using Independence Factor
* `redudant_features.ipynb` Clustermap (Figure 6)
   
### Table 3, Figure 7, and Figure 8
* `audio_annotation.ipynb` code to run experiment/survey
* `analyze_annotations.ipynb` 

### Table 4
* `classification_wo_correlated_features_duration.ipynb` 


### Supplementary Table S1
We removed 24 patients that were recorded using a different device (an iPad). If performance drops significantly, then the original dataset may be using recording set up to dissociate groups (i.e., if features related to iPad are within certain range determined by iPad, then prediction equals patient).
Patients recorded with iPad are: `[3,4,5,8,9,12,13,18,24,27,28,29,31,33,38,53,54,55,56,64,65,66,71,74]`  
* `./data/input/features/egemaps_vector_both_wo-24-patients.csv` dataset
* `./data/output/vfp_v8_wo-24-patients/` pydra-ml scripts
* ```
    egemaps_vector_both_wo-24-patients.csv
    egemaps_vector_speech_wo-24-patients.csv
    egemaps_vector_vowel_wo-24-patients.csv
    ```
    
### Sup. Table S2 and Sup. Table S3

See `test_different_recording.ipynb`







