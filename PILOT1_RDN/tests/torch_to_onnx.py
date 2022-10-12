import torch
from utils import read_config
import os
import pickle
from torchsummary import summary


eval_config = read_config('config.yml', "evaluation")
dot_darts_path, model_name_eval, backtest_start_date, transformer_filename = \
    (os.path.sep).join(eval_config['dot_darts_path'].split('/')), \
    eval_config['model_name'], \
    str(eval_config['backtest_start_date']), \
    str(eval_config['transformer_filename'])

# backtest_end_timestamp = series.time_index[-1]

# use the already defined model name above if no model to be evaluated is defined
model_name = (os.path.sep).join(model_name_eval.split(
    '/')) if model_name_eval is not None else model_name
print('Model to load:', model_name)

# Load best model
# best_model = RNNModel.load_from_checkpoint(work_dir=dot_darts_path, model_name=model_name, best=False)
best_model = torch.load(os.path.join(dot_darts_path, 'checkpoints', model_name, 'checkpoint_model_best_40.pth.tar'), map_location=torch.device('cuda'))
# best_model.device=torch.device('cpu')

# Set path to save evaluation results
models_dir = os.path.join(dot_darts_path, 'checkpoints')
path_to_save_eval = os.path.join(models_dir, model_name)

# Load used scaler
with open(os.path.join(models_dir, model_name, transformer_filename), 'rb') as f:
    transformer_ts= pickle.load(f)

# print(summary(best_model))

# dummy_input = torch.Tensor

# torch.onnx.export(best_model,
#                   dummy_input,
#                   "lstm_120.onnx",
#                   verbose=True,
#                   input_names=input_names,
#                   output_names=output_names,
#                   export_params=True)
print(best_model.state_dict())