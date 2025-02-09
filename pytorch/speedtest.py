import os
import sys
import argparse
import platform
import torch
import numpy as np
import random
import queue
import threading
import time
import OpenImageIO as oiio

from pprint import pprint

def find_and_import_model(models_dir='models', base_name=None, model_name=None, model_file=None):
    """
    Dynamically imports the latest version of a model based on the base name,
    or a specific model if the model name/version is given, and returns the Model
    object named after the base model name.

    :param models_dir: Relative path to the models directory.
    :param base_name: Base name of the model to search for.
    :param model_name: Specific name/version of the model (optional).
    :return: Imported Model object or None if not found.
    """

    import os
    import re
    import importlib

    if model_file:
        module_name = model_file[:-3]  # Remove '.py' from filename to get module name

        print (module_name)

        module_path = f"models.{module_name}"
        module = importlib.import_module(module_path)
        model_object = getattr(module, 'Model')
        return model_object

    # Resolve the absolute path of the models directory
    models_abs_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            models_dir
        )
    )

    # List all files in the models directory
    try:
        files = os.listdir(models_abs_path)
    except FileNotFoundError:
        print(f"Directory not found: {models_abs_path}")
        return None

    # Filter files based on base_name or model_name
    if model_name:
        # Look for a specific model version
        filtered_files = [f for f in files if f == f"{model_name.lower()}.py"]
    else:
        # Find all versions of the model and select the latest one
        # regex_pattern = fr"{base_name}_v(\d+)\.py"
        # versions = [(f, int(m.group(1))) for f in files if (m := re.match(regex_pattern, f))]
        versions = [f for f in files if f.endswith('.py')]
        if versions:
            # Sort by version number (second item in tuple) and select the latest one
            # latest_version_file = sorted(versions, key=lambda x: x[1], reverse=True)[0][0]
            latest_version_file = sorted(versions, reverse=True)[0]
            filtered_files = [latest_version_file]

    # Import the module and return the Model object
    if filtered_files:
        module_name = filtered_files[0][:-3]  # Remove '.py' from filename to get module name
        module_path = f"models.{module_name}"
        module = importlib.import_module(module_path)
        model_object = getattr(module, 'Model')
        return model_object
    else:
        print(f"Model not found: {base_name or model_name}")
        return None



def main():
    parser = argparse.ArgumentParser(description='Speed test.')
    parser.add_argument('--model', type=str, default=None, help='Model name (default: Flownet4_v001_baseline.py)')
    parser.add_argument('--frame_size', type=str, default=None, help='Frame size (default: 4096x1716)')
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')

    args = parser.parse_args()

    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device('cuda')

    if args.frame_size:
        w, h = args.frame_size.split('x')
        h, w = int(h), int(w)
    else:
        h, w = 1716, 4096

    shape = (1, 3, h, w)
    img = torch.randn(shape).to(device)

    model = args.model if args.model else 'Flownet4_v001_baseline'
    Net = find_and_import_model(model_name=model)

    print ('Net info:')
    pprint (Net.get_info())

    net = Net().get_training_model()().to(device)

    import signal
    def create_graceful_exit():
        def graceful_exit(signum, frame):
            '''
            print(f'\nSaving current state to {current_state_dict["trained_model_path"]}...')
            print (f'Epoch: {current_state_dict["epoch"] + 1}, Step: {current_state_dict["step"]:11}')
            torch.save(current_state_dict, current_state_dict['trained_model_path'])
            exit_event.set()  # Signal threads to stop
            process_exit_event.set()  # Signal processes to stop
            '''
            print('\n')
            exit(0)
            # signal.signal(signum, signal.SIG_DFL)
            # os.kill(os.getpid(), signal.SIGINT)
        return graceful_exit
    signal.signal(signal.SIGINT, create_graceful_exit())

    with torch.no_grad():
        while True:
            start_time = time.time()
            flow, mask, conf, merged = net(
                img,
                img,
                # 0.5,
                scale=[8, 4, 2, 1],
                iterations = 1
                )
            # result = result[0].permute(1, 2, 0).numpy(force=True)
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_time = elapsed_time if elapsed_time > 0 else float('inf')
            fps = 1 / elapsed_time
            print (f'\rRunning at {fps:.2f} fps for {shape[3]}x{shape[2]}', end='')

if __name__ == "__main__":
    main()
