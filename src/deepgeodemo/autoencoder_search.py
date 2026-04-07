import itertools
import collections.abc
import pandas as pd
import torch
from .autoencoder_train_latent import train_latent


# Utitlities for generating configurations to search ----------------------

# Split keys in the format 'key: value' into a list
def key_splitter(key):
    return key.split(': ') if ':' in key else [key]

# Generate dictionaries of options from product
# By Seth Johnson via stackoverflow
# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

# Generate configurations based on a base configuration and options
def generate_configs(base, options, state_subversion_of=None):
    configs = []
    subv = 0
    # For each combination of options, create a new configuration
    for option in product_dict(**options):
        description = {**base, **option}
        # Add a version number to the configuration if specified
        if state_subversion_of is not None:
            subv += 1
            description['autoencoder: version'] = str(state_subversion_of) + '-' + str(subv)
        # Create config
        config = {}
        # For each item of the description
        for descr_key, descr_value in description.items():
            descr_key = key_splitter(descr_key)
            config_pointer = config
            # Traverse the config structure to the right position
            for k in descr_key[:-1]:
                if not k in config_pointer.keys():
                    config_pointer[k] = {}
                config_pointer = config_pointer[k]
            # Set the value at the right position
            config_pointer[descr_key[-1]] = descr_value
        # Add the configuration to the list
        configs.append(config)
        # Clean up variables
        del config, description, descr_key, descr_value, config_pointer
    # Return all configs
    return configs

# Flatten dictionaries for reporting
def flatten_dict(d, parent_key = ''):
    items = []
    for k, v in d.items():
        new_key = parent_key + '_' + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping) and v:
            items.extend(flatten_dict(v, new_key).items())
        else:
            # Handle tensors
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    v = v.detach().cpu().tolist()
            v = v if isinstance(v, (int, float)) else str(v)
            items.append((new_key, v))
    return dict(items)

# Generate final report from configurations and reports
def generate_final_report(configs, reports):
    flat_configs = [flatten_dict(config) for config in configs]
    flat_reports = [flatten_dict(report) for report in reports]
    final_reports = []
    for config, report in zip(flat_configs, flat_reports):
        final_reports.append({**config, **report})
    final_report_df = pd.DataFrame(final_reports)
    return final_report_df


# Main method searching across configurations -----------------------------

def explore_configs(ae_base, ae_options, state_subversion_of=0, create_latent=False, verbose=True):
    # Generate configurations based on the base and options
    ae_configs = generate_configs(ae_base, ae_options, state_subversion_of)
    ae_reports = []
    # Run the training for each configuration
    for i, ae_config in enumerate(ae_configs):
        print(f'\n\n{'-'*76}')
        print(f'Exploring configuration {i+1} of {len(ae_configs)}')
        print(f'{'-'*76}\n')
        ae_report = train_latent(ae_config, create_latent=create_latent, verbose=verbose)
        ae_reports.append(ae_report)
        del ae_report
    ae_final_report = generate_final_report(ae_configs, ae_reports)
    return ae_final_report