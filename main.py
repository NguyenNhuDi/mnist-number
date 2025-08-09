import torch 


if __name__ == '__main__':
    device_name = torch.cuda.get_device_name(0)

    print(f'Current device: {device_name}')