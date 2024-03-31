import pandas as pd
import os


def read_logs_to_df(logs_dir):
    data = []

    for model_dir in os.listdir(logs_dir):
        model_path = os.path.join(logs_dir, model_dir)
        if os.path.isdir(model_path):
            for log in ['training_log.csv', 'validation_log.csv']:
                file_path = os.path.join(model_path, log)
                if os.path.exists(file_path):
                    tmp = pd.read_csv(file_path)
                    tmp['Model'] = model_dir
                    tmp['Type'] = 'Training' if 'training' in log else 'Validation'

                    data.append(tmp)
    return pd.concat(data, ignore_index=True)
