import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from darts.metrics import mape as mape_darts


def darts_single_block_forecast(model, block_n_steps, series, future_covariates, past_covariates):

    pred = model.predict(n=block_n_steps,
                         future_covariates=future_covariates,
                         past_covariates=past_covariates,
                         series=series)
    return pred


def darts_block_n_step_ahead_forecast(model,
                                      history,
                                      test,
                                      block_n_steps=24,
                                      n_blocks=31,
                                      future_covariates=None,
                                      past_covariates=None,
                                      path_to_save_eval=None):
    """ This function produces a chained darts forecast that is a forecast in successive blocks. 
    Thus, in the first iteration a forecast of length block_n_steps is produced using the darts "predict" method. 
    At each next of the "n_blocks" iterations the predict function is called but fed with the ground truth historical 
    data of the previous block  as input timeseries. This helps to avoid lengthy forecasts that are produced 
    without updating with newly obtained ground truth, as this not a realistic condition for an online power system
    and surely leads to undesirable error propagations. Append whatever you like as historical news (no need to be 
    adjacent to training set but should be longer than input_chunk_length)... However, history and test must be adjacent 
    time series (if test exists). And future and past covariates must be taken care of -> need to be provided.
    
    Parameters
    ----------
        model: darts.models.forecasting.[darts_forecasting_model_class].[darts_model_name]Model
            A darts model providing a .predict method.

        history: darts.timeseries.TimeSeries
            A darts timeseries dataset that carries the initial historical values of the timeseries.
        
        test: darts.timeseries.TimeSeries
            A darts timeseries dataset that carries the unknown values needed to evaluate the model
            
        block_n_steps: int
            The number of timesteps (forecast horizon) of each block forecast. It implies the number of
            timesteps after which historical ground truth values of the target timeseries are fed back
            to the model so as to be included ois renewed for the model.
        
        n_blocks: int
            The number of forecast blocks. Multiplied by block_n_steps it results to the total forecasting
            horizon.

        future_covariates: Optional[darts.timeseries.TimeSeries] 
            Optionally, a series or sequence of series specifying future-known covariates (see darts docs).
        
        past_covariates: Optional[darts.timeseries.TimeSeries] 
            Optionally, a series or sequence of series specifying past-observed covariates (see darts docs)

    Returns
    ----------

    """

    # you can always feed your time series history to the model
    # or at least a history longer or equal to input chunk length
    if test is not None:
        series = history.append(test)
    elif n_blocks != 1:
        n_blocks = 1
        message = 'Warning: n_blocks was set equal to 1 as no test set was provided to further update history. One forecasting iteration will be run.'
        print(message)
        logging.info(message)

    # calculate predictions in blocks, updating the history after each block
    for i in tqdm(range(n_blocks)):
        pred_i = darts_single_block_forecast(model,
                                             block_n_steps,
                                             history,
                                             future_covariates,
                                             past_covariates)
        pred = pred_i if i == 0 else pred.append(pred_i)
        if test is None:
            return pred
        history = history.append(test[block_n_steps*(i):block_n_steps*(i+1)])

    # evaluate
    plt.figure(figsize=(15, 8))
    # series.drop_before(pd.Timestamp(pred.time_index[0] - datetime.timedelta(
    #     days=7))).drop_after(pred.time_index[-1]).plot(label='actual')
    if len(series) - len(test) - 7*24 > 0:
        series.drop_before(len(series) - len(test) - 7 *
                           24).drop_after(pred.time_index[-1]).plot(label='actual')
    else:
        series.drop_after(pred.time_index[-1]).plot(label='actual')

    pred.plot(label='forecast')
    plt.legend()
    mape_error = mape_darts(test, pred)
    print('MAPE = {:.2f}%'.format(mape_error))

    if path_to_save_eval is not None:
        plt.savefig(os.path.join(path_to_save_eval,
                    f"block_n_steps_{block_n_steps}_n_blocks_{n_blocks}_mape_{mape_error:.2f}.png"))

    return mape_error, pred
