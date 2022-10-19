import logging

import torch


def get_device_info():
    device_info = {}
    dummy_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_info['device_type'] = dummy_device.type
    device_info['device_count'] = torch.cuda.device_count()
    device_info['device_name'] = torch.cuda.get_device_name(dummy_device)
    return device_info
