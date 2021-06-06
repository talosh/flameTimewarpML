import os
import sys
import psutil
import importlib

# Constant values
INPUT_QUEUE_SIZE = 96
OUTPUT_QUEUE_SIZE = 9


def safe_threads_number(h, w, print_info=True):
    import multiprocessing as mp

    max_cpu_workers = mp.cpu_count() - 2
    available_ram = psutil.virtual_memory()[1] / (1024 ** 3)
    megapixels = (h * w) / (10 ** 6)

    if sys.platform == 'darwin':
        thread_ram = megapixels * 0.99
    else:
        thread_ram = megapixels * 0.99

    sim_workers = round(available_ram / thread_ram)

    if sim_workers < 1:
        sim_workers = 1
    elif sim_workers > max_cpu_workers:
        sim_workers = max_cpu_workers

    if print_info:
        print('---\nFree RAM: %s Gb available' % '{0:.1f}'.format(available_ram))
        print('Image size: %s x %s' % (w, h))
        print('Peak memory usage estimation: %s Gb per CPU thread ' % '{0:.1f}'.format(thread_ram))
        if sim_workers == max_cpu_workers:
            print('Using %s CPU worker thread%s (of %s available)' % (sim_workers, '' if sim_workers == 1 else 's', mp.cpu_count()))
        else:
            print('Limiting therads to %s CPU worker thread%s (of %s available) to prevent RAM from overflow\n---' % (sim_workers, '' if sim_workers == 1 else 's', mp.cpu_count()))
        if thread_ram > available_ram:
            print('Warning: estimated peak memory usage is greater then RAM avaliable')

    return sim_workers, thread_ram


def load_model(model_path, cpu=False):
    model_names = [
        'RIFE_HD',
        'RIFE_HDv2',
        'RIFE_HDv3'
        ]

    models_list = []
    for model_name in model_names:
        if cpu:
            models_list.append('model_cpu.' + model_name)
        else:
            models_list.append('model.' + model_name)

    for model_name in sorted(models_list, reverse=True):
        try:
            rife = importlib.import_module(model_name)
            model = rife.Model()
            model.load_model(model_path, -1)
            return model
        except:
            print ('failed to load %s' % model_name)

    
