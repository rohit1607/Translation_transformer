import os

data_path = "/media/HDD/rohit/Translation_transformer/my-translat-transformer/data/GPT_dset_DG3/static_obs/"

dat_names = []

def get_names(data_path=""):
    global dat_names
    for path in os.listdir(data_path):
        if (path[46:65] == 'multi_ran_stat_new_' and (path[-1]=='0' or path[-1]=='1'or path[-1]=='2' or path[-1]=='3'or path[-1]=='4' or path[-1]=='5' or path[-1]=='6' or path[-1]=='7' or path[-1]=='8' or path[-1]=='9')):
            dat_names.append(data_path+path+"/")
    return dat_names


if __name__=="__main__":
    x = get_names(data_path)
    print(len(x))
    print(x)