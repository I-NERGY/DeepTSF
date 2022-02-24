1. Reproduce conda environment using the conda.yml file.
3. Activate conda environment
3. Run:
(May need to replace future-covs-uri with past-covs-uri depending on the provided artifacts or remove it entirely.
Also need to replace paths depending on the directory you extracted the zip)

 python .\evaluate_forecasts.py --mode "local" --series-uri ".\models\nbeats\artifacts\features\series.csv" --scaler-uri ".\models\nbeats\artifacts\scalers\scaler_series.pkl" --setup-uri ".\models\nbeats\artifacts\features\split_info.yml" --model-uri ".\models\nbeats\artifacts\checkpoints\model_best.pth.tar"  --past-covs-uri ".\models\nbeats\artifacts\features\past_covariates_transformed.csv"

In case remote uris need to be provided change --mode to "remote".