#import torch
from bondnet.scripts.train_transfer import train_transfer
from glob import glob


def main():
    files = glob("settings*.txt")
    for file in files:
        print(file)
        train_transfer(file)
main()


# Todo implement a job for each device but more on that later
#devices = [torch.device("cuda:" + str(i)) for i in range(torch.cuda.device_count())]
