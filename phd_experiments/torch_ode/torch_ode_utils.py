import datetime
import torch


def get_device_info():
    device_info = {}
    dummy_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_info['device_type'] = dummy_device.type
    device_info['device_count'] = torch.cuda.device_count()
    device_info['device_name'] = torch.cuda.get_device_name(dummy_device)
    return device_info


def format_timedelta(time_delta: datetime.timedelta):
    tot_seconds = time_delta.seconds
    hours = tot_seconds // 3600
    minutes = (tot_seconds // 60) % 60
    seconds = tot_seconds - hours * 3600 - minutes * 60
    return f"{hours} hours , {minutes} minutes, {seconds} seconds"
