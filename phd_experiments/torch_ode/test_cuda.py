import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())
    cuda_device = torch.device('cuda')
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
