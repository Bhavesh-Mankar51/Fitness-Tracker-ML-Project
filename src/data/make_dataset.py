import pandas as pd
from glob import glob
import re
import os

# single_file_acc = pd.read_csv(
#     "/Users/bhaveshmankar/data-science-template/data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")

# single_file_gyr = pd.read_csv(
#     "/Users/bhaveshmankar/data-science-template/data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")


files = glob(
    "/Users/bhaveshmankar/data-science-template/data/raw/MetaMotion/*.csv")

# data_path = "/Users/bhaveshmankar/data-science-template/data/raw/MetaMotion"



def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        filename = os.path.basename(f)
        parts = filename.split("-")

        participant = parts[0]
        label = parts[1]
        category = re.sub(r'\d+', '', parts[2].split("_")[0])

        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)


data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

# 1. Renames columns (make sure they match the existing DataFrame)
data_merged.columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "participant", "label", "category", "set"]



sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y":    "mean",
    "gyr_z":    "mean",
     "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last"
}


data_merged[:1000].resample(rule="200ms").apply(sampling)

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled["set"] = data_resampled["set"].astype("int")
data_resampled.info()
data_resampled[:10]
data_resampled.to_pickle("/Users/bhaveshmankar/data-science-template/data/interim/01_data_processed.pkl")