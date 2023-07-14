import os

data_path = "/home/rohit/Documents/Research/data_prep/HDD_data/GenHW/"

dat_names = []

def get_names(data_path=""):
    global dat_names
    for path in os.listdir(data_path):
        dat_names.append(data_path+path+"/")
    return dat_names


if __name__=="__main__":
    x = get_names(data_path)
    print(len(x))
    print(x)