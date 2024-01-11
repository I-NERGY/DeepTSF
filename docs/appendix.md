# MLproject file

The MLproject file describes the way the ML pipeline is executed. It is
in YAML format, and it defines the name of the MLflow project (in this
case DeepTSF_workflow), the file to use to build the environment the
user desires to work with (in this case conda.yaml), and the entry
points of our project.

Each entry point corresponds to a specific stage of the pipeline
describing the respective python command alongside its parameters. In
this case, each entry point runs the main python file for the stage it
corresponds to, and passes the parameters to the file using the Click
library . The file is shown below:

``` yaml
name: DeepTSF_workflow

conda_env: ../conda.yaml

entry_points:

  load_raw_data:
    parameters:
      series_csv: {type: str, default: None}
      series_uri: {type: str, default: None}
      past_covs_csv: {type: str, default: None}
      past_covs_uri: {type: str, default: None}
      future_covs_csv: {type: str, default: None}
      future_covs_uri: {type: str, default: None}
      day_first: {type: str, default: "true"}
      multiple: {type: str, default: "false"}
      resolution: {type: str, default: None}
      from_database: {type: str, default: "false"}
      database_name: {type: str, default: "rdn_load_data"}

    command: |
      python load_raw_data.py --series-csv {series_csv} --series-uri {series_uri} --day-first {day_first} --multiple {multiple} --resolution {resolution} --from-database {from_database} --database-name {database_name} --past-covs-csv {past_covs_csv} --past-covs-uri {past_covs_uri} --future-covs-csv {future_covs_csv} --future-covs-uri {future_covs_uri} 


  etl:
    parameters:
      series_csv: {type: str, default: None}
      series_uri: {type: str, default: None}
      resolution: {type: str, default: None}
      year_range: {type: str, default: None}
      time_covs: {type: str, default: "false"}
      day_first: {type: str, default: "true"}
      country: {type: str, default: "PT"}
      std_dev: {type: str, default: 4.5}
      max_thr: {type: str, default: -1}
      a: {type: str, default: 0.3}
      wncutoff: {type: str, default: 0.000694}
      ycutoff: {type: str, default: 3}
      ydcutoff: {type: str, default: 30}
      multiple: {type: str, default: "false"}
      l_interpolation: {type: str, default : "false"}
      rmv_outliers: {type: str, default: "true"}
      convert_to_local_tz: {type: str, default: "true"}
      ts_used_id: {type: str, default: None}
      infered_resolution_series: {type: str, default: "15"}
      min_non_nan_interval: {type: str, default: "24"}
      cut_date_val: {type: str, default: None}
      infered_resolution_past: {type: str, default: "15"} 
      past_covs_csv: {type: str, default: None} 
      past_covs_uri: {type: str, default: None}
      infered_resolution_future: {type: str, default: "15"} 
      future_covs_csv: {type: str, default: None} 
      future_covs_uri: {type: str, default: None}
      resampling_agg_method: {type: str, default: "averaging"}
    command: |
      python etl.py --series-csv {series_csv} --series-uri {series_uri} --resolution {resolution} --year-range {year_range} --time-covs {time_covs} --day-first {day_first} --country {country} --std-dev {std_dev} --max-thr {max_thr} --a {a} --wncutoff {wncutoff} --ycutoff {ycutoff} --ydcutoff {ydcutoff} --multiple {multiple} --l-interpolation {l_interpolation} --rmv-outliers {rmv_outliers} --convert-to-local-tz {convert_to_local_tz} --ts-used-id {ts_used_id} --infered-resolution-series {infered_resolution_series} --min-non-nan-interval {min_non_nan_interval} --cut-date-val {cut_date_val} --past-covs-csv {past_covs_csv} --past-covs-uri {past_covs_uri} --future-covs-csv {future_covs_csv} --future-covs-uri {future_covs_uri} --infered-resolution-past {infered_resolution_past}  --past-covs-csv {past_covs_csv} --past-covs-uri {past_covs_uri} --infered-resolution-future {infered_resolution_future} --future-covs-csv {future_covs_csv} --future-covs-uri {future_covs_uri} --resampling-agg-method {resampling_agg_method}

  train:
    parameters:
      series_csv: {type: str, default: None}
      series_uri: {type: str, default: None}
      future_covs_csv: {type: str, default: None}
      future_covs_uri: {type: str, default: None}
      past_covs_csv: {type: str, default: None}
      past_covs_uri: {type: str, default: None}
      cut_date_val: {type: str, default: None}
      cut_date_test: {type: str, default: None}
      test_end_date: {type: str, default: None}
      darts_model: {type: str, default: None}
      device: {type: str, default: gpu}
      hyperparams_entrypoint: {type: str, default: None}
      scale: {type: str, default: "true"}
      scale_covs: {type: str, default: "true"}
      multiple: {type: str, default: "false"}
      training_dict: {type: str, default: None}
      num_workers: {type: str, default: 4}
      day_first: {type: str, default: "true"}
      resolution: {type: str, default: None}

    command: |
      python ../training.py --series-csv {series_csv} --series-uri {series_uri} --future-covs-csv {future_covs_csv} --future-covs-uri {future_covs_uri} --past-covs-csv {past_covs_csv}  --past-covs-uri {past_covs_uri} --cut-date-val {cut_date_val} --cut-date-test {cut_date_test} --test-end-date {test_end_date} --darts-model {darts_model} --device {device} --hyperparams-entrypoint {hyperparams_entrypoint} --scale {scale} --scale-covs {scale_covs} --multiple {multiple} --training-dict {training_dict} --cut-date-val {cut_date_val} --num-workers {num_workers} --day-first {day_first} --resolution {resolution}

  eval:
    parameters:
      mode: {type: str, default: remote}
      series_uri: {type: str, default: None}
      future_covs_uri: {type: str, default: None}
      past_covs_uri: {type: str, default: None}
      scaler_uri: {type: str, default: None}
      cut_date_test: {type: str, default: None}
      test_end_date: {type: str, default: None}
      model_uri: {type: str, default: None}
      model_type: {type: str, default: pl}
      forecast_horizon: {type: str, default: None}
      stride: {type: str, default: None}
      retrain: {type: str, default: "false"}
      shap_input_length: {type: str, default: None}
      shap_output_length: {type: str, default: None}
      size: {type: str, default: 10}
      analyze_with_shap: {type: str, default: "false"}
      multiple: {type: str, default: "false"}
      eval_series: {type: str, default: None}
      cut_date_val: {type: str, default: None}
      day_first: {type: str, default: "true"}
      resolution: {type: str, default: None}
      eval_method: {type: str, default: "ts_ID"}
      evaluate_all_ts: {type: str, default: "false"}
      m_mase: {type: str, default: "1"}
      num_samples: {type: str, default: "1"}

    command: |
      python ../evaluate_forecasts.py --mode {mode} --series-uri {series_uri} --future-covs-uri {future_covs_uri} --model-type {model_type} --past-covs-uri {past_covs_uri} --scaler-uri {scaler_uri} --cut-date-test {cut_date_test} --test-end-date {test_end_date} --model-uri {model_uri} --forecast-horizon {forecast_horizon} --stride {stride} --retrain {retrain} --shap-input-length {shap_input_length} --shap-output-length {shap_output_length} --size {size} --analyze-with-shap {analyze_with_shap} --multiple {multiple} --eval-series {eval_series} --cut-date-val {cut_date_val} --day-first {day_first} --resolution {resolution} --eval-method {eval_method} --evaluate-all-ts {evaluate_all_ts} --m-mase {m_mase} --num-samples {num_samples}


  optuna_search:
    parameters:
      series_csv: {type: str, default: None}
      series_uri: {type: str, default: None}
      future_covs_csv: {type: str, default: None}
      future_covs_uri: {type: str, default: None}
      past_covs_csv: {type: str, default: None}
      past_covs_uri: {type: str, default: None}
      resolution: {type: str, default: None}
      year_range: {type: str, default: None}
      darts_model: {type: str, default: None}
      hyperparams_entrypoint: {type: str, default: None}
      cut_date_val: {type: str, default: None}
      cut_date_test: {type: str, default: None}
      test_end_date: {type: str, default: None}
      device: {type: str, default: gpu}
      forecast_horizon: {type: str, default: None}
      stride: {type: str, default: None}
      retrain: {type: str, default: false}
      scale: {type: str, default: "true"}
      scale_covs: {type: str, default: "true"}
      multiple: {type: str, default: "false"}
      eval_series: {type: str, default: None}
      n_trials: {type: str, default: 100}
      num_workers: {type: str, default: 4}
      day_first: {type: str, default: "true"}
      eval_method: {type: str, default: "ts_ID"}
      loss_function: {type: str, default: "mape"}
      evaluate_all_ts: {type: str, default: "false"}
      grid_search: {type: str, default: "false"}
      num_samples: {type: str, default: "1"}

    command: |
      python ../optuna_search.py --series-csv {series_csv} --series-uri {series_uri} --future-covs-csv {future_covs_csv} --future-covs-uri {future_covs_uri} --past-covs-csv {past_covs_csv}  --past-covs-uri {past_covs_uri} --resolution {resolution} --year-range {year_range} --darts-model {darts_model} --hyperparams-entrypoint {hyperparams_entrypoint} --cut-date-val {cut_date_val} --cut-date-test {cut_date_test} --test-end-date {test_end_date} --device {device} --forecast-horizon {forecast_horizon} --stride {stride} --retrain {retrain} --scale {scale} --scale-covs {scale_covs} --multiple {multiple} --eval-series {eval_series} --n-trials {n_trials} --num-workers {num_workers} --day-first {day_first} --eval-method {eval_method} --loss-function {loss_function} --evaluate-all-ts {evaluate_all_ts} --grid-search {grid_search} --num-samples {num_samples}



  exp_pipeline:
    parameters:
      series_csv: {type: str, default: None}
      series_uri: {type: str, default: None}
      past_covs_csv: {type: str, default: None}
      past_covs_uri: {type: str, default: None}
      future_covs_csv: {type: str, default: None}
      future_covs_uri: {type: str, default: None}
      resolution: {type: str, default: None}
      year_range: {type: str, default: None}
      time_covs: {type: str, default: "false"}
      hyperparams_entrypoint: {type: str, default: None}
      cut_date_val: {type: str, default: None}
      cut_date_test: {type: str, default: None}
      test_end_date: {type: str, default: None}
      darts_model: {type: str, default: None}
      device: {type: str, default: gpu}
      forecast_horizon: {type: str, default: None}
      stride: {type: str, default: None}
      retrain: {type: str, default: false}
      ignore_previous_runs: {type: str, default: "true"}
      scale: {type: str, default: "true"}
      scale_covs: {type: str, default: "true"}
      day_first: {type: str, default: "true"}
      country: {type: str, default: "PT"}
      std_dev: {type: str, default: 4.5}
      max_thr: {type: str, default: -1}
      a: {type: str, default: 0.3}
      wncutoff: {type: str, default: 0.000694}
      ycutoff: {type: str, default: 3}
      ydcutoff: {type: str, default: 30}
      shap_data_size: {type: str, default: 100}
      analyze_with_shap: {type: str, default: false}
      multiple: {type: str, default: "false"}
      eval_series: {type: str, default: None}
      n_trials: {type: str, default: 100}
      opt_test: {type: str, default: "false"}
      from_database: {type: str, default: "false"}
      database_name: {type: str, default: "rdn_load_data"}
      num_workers: {type: str, default: 4}
      eval_method: {type: str, default: "ts_ID"}
      l_interpolation: {type: str, default : "false"}
      rmv_outliers: {type: str, default: "true"}
      loss_function: {type: str, default: "mape"}
      evaluate_all_ts: {type: str, default: "false"}
      convert_to_local_tz: {type: str, default: "true"}
      grid_search: {type: str, default: "false"}
      shap_input_length: {type: str, default: None}
      ts_used_id: {type: str, default: None}
      m_mase: {type: str, default: "1"}
      min_non_nan_interval: {type: str, default: "24"}
      num_samples: {type: str, default: "1"}
      resampling_agg_method: {type: str, default: "averaging"}

    command: |
      python ../experimentation_pipeline.py --series-csv {series_csv} --series-uri {series_uri} --resolution {resolution} --year-range {year_range} --time-covs {time_covs} --cut-date-val {cut_date_val} --cut-date-test {cut_date_test} --test-end-date {test_end_date} --darts-model {darts_model} --device {device} --hyperparams-entrypoint {hyperparams_entrypoint} --forecast-horizon {forecast_horizon} --stride {stride} --retrain {retrain} --ignore-previous-runs {ignore_previous_runs} --scale {scale} --scale-covs {scale_covs} --day-first {day_first} --country {country} --std-dev {std_dev} --max-thr {max_thr} --a {a} --wncutoff {wncutoff} --ycutoff {ycutoff} --ydcutoff {ydcutoff} --shap-data-size {shap_data_size} --analyze-with-shap {analyze_with_shap} --multiple {multiple} --eval-series {eval_series} --n-trials {n_trials} --opt-test {opt_test} --from-database {from_database} --database-name {database_name} --num-workers {num_workers} --eval-method {eval_method} --l-interpolation {l_interpolation} --rmv-outliers {rmv_outliers} --loss-function {loss_function} --evaluate-all-ts {evaluate_all_ts} --convert-to-local-tz {convert_to_local_tz} --grid-search {grid_search} --shap-input-length {shap_input_length} --ts-used-id {ts_used_id} --m-mase {m_mase} --min-non-nan-interval {min_non_nan_interval} --past-covs-csv {past_covs_csv} --past-covs-uri {past_covs_uri} --future-covs-csv {future_covs_csv} --future-covs-uri {future_covs_uri} --num-samples {num_samples} --resampling-agg-method {resampling_agg_method}


  inference:
    parameters:
      pyfunc_model_folder: {type: str, default: s3://mlflow-bucket/2/33d85746285c42a7b3ef403eb2f5c95f/artifacts/pyfunc_model}
      forecast_horizon:  {type: str, default: None}
      series_uri: {type: str, default: None}
      past_covariates_uri: {type: str, default: None}
      future_covariates_uri: {type: str, default: None}
      roll_size: {type: str, default: 96}
      batch_size:  {type: str, default: 1}

    command: |
      python ../inference.py --pyfunc-model-folder {pyfunc_model_folder} --forecast-horizon {forecast_horizon} --series-uri {series_uri} --past-covariates-uri {past_covariates_uri} --future-covariates-uri {future_covariates_uri} --roll-size {roll_size} --batch-size {batch_size}
      
```

# Forecasting back-end architecture

The architecture of the forecasting back-end is illustrated below
<a href="#fig:mlflow_architecture" data-reference-type="ref"
data-reference="fig:mlflow_architecture"></a>:

<figure id="fig:mlflow_architecture">
<img src="docs/MLflow_Architecture_Diagram.png" />
</figure>

The components that run on the MLflow server host are responsible for
the storage of the models and of other relevant artifacts (MinIO model /
artifact store container), for the logging of parameters and metrics of
each step (tracking database), as well as for running the MLflow user
interface presented to the user (MLflow tracking container). The
component on the left (which runs on the MLflow client host) is the
executor the DeepTSF ML pipelines which acts as a client to the services
of the MLflow server. All these components are containerized using
Docker and can either run on the same ore different machines.

# Allowed file formats

The format of the csv files DeepTSF can accept depends on the nature of
the problem it is trying to solve. More specifically, in case of a
single time series file, its format is illustrated in Table
<a href="#tab:csv_simple" data-reference-type="ref"
data-reference="tab:csv_simple">1</a>:

| Datetime            | Value |
|:--------------------|:------|
| 2015-04-09 00:00:00 | 7893  |
| 2015-04-09 01:00:00 | 8023  |
| 2015-04-09 02:00:00 | 8572  |
| ...                 | ...   |

Single time series file format in hourly resolution

<span id="tab:csv_simple" label="tab:csv_simple"></span>

In this table, the Datetime column simply stores the dates and times of
each observation, and the Value column stores the value that has been
observed.

If we are solving a multiple and / or multivariate time series problem,
then the file format (along with example values) is shown in Table
<a href="#tab:csv_multi" data-reference-type="ref"
data-reference="tab:csv_multi">2</a>:

| Index | Date       | ID  | Timeseries ID | 00:00:00 | ... |     |     |
|:------|:-----------|:----|:--------------|:---------|:----|:----|:----|
| 0     | 2015-04-09 | PT  | PT            | 5248     | ... |     |     |
| 1     | 2015-04-09 | ES  | ES            | 25497    | ... |     |     |
| ...   | ...        | ... | ...           | ...      | ... |     |     |

Multiple and / or multivariate time series file format

<span id="tab:csv_multi" label="tab:csv_multi"></span>

The columns that can be present in the csv have the following meaning:

-   Index: Simply a monotonic integer range

-   Date: The Date each row is referring to

-   ID: Each ID corresponds to a component of a time series in the file.
    This ID must be unique for each time series component in the file.
    If referring to country loads it can be the country code. In this
    case, this will be used to obtain the country holidays for the
    imputation function as well as the time covariates.

-   Timeseries ID (Optional): Timeseries ID column is not compulsory,
    and shows the time series to which each component belongs. If
    Timeseries ID is not present, it is assumed that each component
    represents one separate series (the column is set to ID).

-   Time columns: Columns that store the Value of each component. They
    must be consecutive and separated by resolution minutes. They should
    start at 00:00:00, and end at 24:00:00 - resolution

The checks that are performed when validating a file are the following:

For all time series:

-   The dataframe can not be empty

-   All the dates must be sorted

For non-multiple time series:

-   Column Datetime must be used as an index

-   If the time series is the main dataset, Load must be the only other
    column in the dataframe

-   If the time series is a covariates time series, there must be only
    one column in the dataframe named arbitrarily

For multiple timeseries:

-   Columns Date, ID, and the time columns exist in any order

-   Only the permitted column names exist in the dataframe (see Multiple
    timeseries file format bellow)

-   All timeseries in the dataframe have the same number of components

For more information about these files see the documentation .

# Providing the hyperparameters to DeepTSF

To use DeepTSF’s optimization mechanism the user needs to provide the
desired hyperparameter grid in the config_opt.yml file using the YAML
format , as shown in Fig.
<a href="#fig:hyperparams" data-reference-type="ref"
data-reference="fig:hyperparams">2</a>. This is the grid used for the
example of <a href="#sec:3:CLI" data-reference-type="ref"
data-reference="sec:3:CLI">[sec:3:CLI]</a>. In the YAML file, the
possible values for each hyperparameter need to be given in a list
format as follows:

-   Format \["range", start, end, step\]: a list of hyperparameter
    values are considered ranging from value "start" till "end" with the
    step being defined by the last value of the list.

-   Format \["list", value_1, ..., value_n\]: All the listed parameters
    ({value_1, ..., value_n}) are considered in the grid.

More information is included in the main documentation of DeepTSF.

<figure id="fig:hyperparams">
<div class="sourceCode" id="cb1" data-fontsize="\footnotesize"
data-linenos="False" data-frame="lines" data-framesep="2mm"><pre
class="sourceCode yaml"><code class="sourceCode yaml"><span id="cb1-1"><a href="#cb1-1" aria-hidden="true" tabindex="-1"></a><span class="fu">NBEATS_example</span><span class="kw">:</span></span>
<span id="cb1-2"><a href="#cb1-2" aria-hidden="true" tabindex="-1"></a><span class="at">    </span><span class="fu">input_chunk_length</span><span class="kw">:</span><span class="at"> </span><span class="kw">[</span><span class="st">&quot;range&quot;</span><span class="kw">,</span><span class="at"> </span><span class="dv">48</span><span class="kw">,</span><span class="at"> </span><span class="dv">240</span><span class="kw">,</span><span class="at"> </span><span class="dv">24</span><span class="kw">]</span></span>
<span id="cb1-3"><a href="#cb1-3" aria-hidden="true" tabindex="-1"></a><span class="at">    </span><span class="fu">output_chunk_length</span><span class="kw">:</span><span class="at"> </span><span class="dv">24</span></span>
<span id="cb1-4"><a href="#cb1-4" aria-hidden="true" tabindex="-1"></a><span class="at">    </span><span class="fu">num_stacks</span><span class="kw">:</span><span class="at"> </span><span class="kw">[</span><span class="st">&quot;range&quot;</span><span class="kw">,</span><span class="at"> </span><span class="dv">1</span><span class="kw">,</span><span class="at"> </span><span class="dv">10</span><span class="kw">,</span><span class="at"> </span><span class="dv">1</span><span class="kw">]</span></span>
<span id="cb1-5"><a href="#cb1-5" aria-hidden="true" tabindex="-1"></a><span class="at">    </span><span class="fu">num_blocks</span><span class="kw">:</span><span class="at"> </span><span class="kw">[</span><span class="st">&quot;range&quot;</span><span class="kw">,</span><span class="at"> </span><span class="dv">1</span><span class="kw">,</span><span class="at"> </span><span class="dv">10</span><span class="kw">,</span><span class="at"> </span><span class="dv">1</span><span class="kw">]</span></span>
<span id="cb1-6"><a href="#cb1-6" aria-hidden="true" tabindex="-1"></a><span class="at">    </span><span class="fu">num_layers</span><span class="kw">:</span><span class="at"> </span><span class="kw">[</span><span class="st">&quot;range&quot;</span><span class="kw">,</span><span class="at"> </span><span class="dv">1</span><span class="kw">,</span><span class="at"> </span><span class="dv">5</span><span class="kw">,</span><span class="at"> </span><span class="dv">1</span><span class="kw">]</span></span>
<span id="cb1-7"><a href="#cb1-7" aria-hidden="true" tabindex="-1"></a><span class="at">    </span><span class="fu">generic_architecture</span><span class="kw">:</span><span class="at"> </span><span class="ch">True</span></span>
<span id="cb1-8"><a href="#cb1-8" aria-hidden="true" tabindex="-1"></a><span class="at">    </span><span class="fu">layer_widths</span><span class="kw">:</span><span class="at"> </span><span class="dv">64</span></span>
<span id="cb1-9"><a href="#cb1-9" aria-hidden="true" tabindex="-1"></a><span class="at">    </span><span class="fu">expansion_coefficient_dim</span><span class="kw">:</span><span class="at"> </span><span class="dv">5</span></span>
<span id="cb1-10"><a href="#cb1-10" aria-hidden="true" tabindex="-1"></a><span class="at">    </span><span class="fu">n_epochs</span><span class="kw">:</span><span class="at"> </span><span class="dv">300</span></span>
<span id="cb1-11"><a href="#cb1-11" aria-hidden="true" tabindex="-1"></a><span class="at">    </span><span class="fu">random_state</span><span class="kw">:</span><span class="at"> </span><span class="dv">0</span></span>
<span id="cb1-12"><a href="#cb1-12" aria-hidden="true" tabindex="-1"></a><span class="at">    </span><span class="fu">nr_epochs_val_period</span><span class="kw">:</span><span class="at"> </span><span class="dv">2</span></span>
<span id="cb1-13"><a href="#cb1-13" aria-hidden="true" tabindex="-1"></a><span class="at">    </span><span class="fu">batch_size</span><span class="kw">:</span><span class="at"> </span><span class="kw">[</span><span class="st">&quot;list&quot;</span><span class="kw">,</span><span class="at"> </span><span class="dv">256</span><span class="kw">,</span><span class="at"> </span><span class="dv">512</span><span class="kw">,</span><span class="at"> </span><span class="dv">1024</span><span class="kw">,</span><span class="at"> </span><span class="dv">1280</span><span class="kw">,</span><span class="at"> </span><span class="dv">1536</span><span class="kw">,</span><span class="at"> </span><span class="dv">2048</span><span class="kw">]</span></span></code></pre></div>
</figure>
