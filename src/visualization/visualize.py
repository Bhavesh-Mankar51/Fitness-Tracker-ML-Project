import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


df = pd.read_pickle(
    "/Users/bhaveshmankar/data-science-template/data/interim/01_data_processed.pkl")

set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"].reset_index(drop=True))

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

category_df = df.query("label == 'squat'").query(
    "participant == 'A'").reset_index()

fix, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

participant_df = df.query("label == 'bench'").sort_values(
    "participant").reset_index()

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


# Plot multiple axis

label = "squat"
participant = "A"
all_axis_df = df.query(f"label == '{label}'").query(
    f"participant == '{participant}'").reset_index()

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'").query(
                f"participant == '{participant}'").reset_index()
        )

        if len(all_axis_df) > 0:

            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()


for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'").query(
                f"participant == '{participant}'").reset_index()
        )

        if len(all_axis_df) > 0:

            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()

label = "row"
participant = "A"
combined_plot_df = (
    df.query(f"label == '{label}'").query(
        f"participant == '{participant}'").reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=True, shadow=True)


ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=True, shadow=True)

ax[1].set_xlabel("samples")

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}'").query(
                f"participant == '{participant}'").reset_index()
        )

        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=True, shadow=True)
            ax[1].set_xlabel("samples")
            plt.show()
            
            plt.savefig(f"/Users/bhaveshmankar/data-science-template/reports/figures/{label.title()}({participant}).png")