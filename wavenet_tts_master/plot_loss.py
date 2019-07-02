import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TITLE = "WaveNet TTS: conditioning on phoneme sequence"

# LOGDIR = "/home/cnsl/TTS/wavenet_tts/wavenet_tts_master/logdir/train/1. repeat_mono/"
# LOG_FILE = "repeat_mono.log"

LOGDIR = "/home/cnsl/TTS/wavenet_tts/wavenet_tts_master/logdir/train/14.avg_repeat_dil128ch_36"
LOG_FILE = "14.avg_repeat_dil128ch_36.log"
# LOGDIR = "/home/cnsl/TTS/wavenet_tts/wavenet_tts_master/logdir/train/5.repeat_tri_lc_causal_layer"
# LOG_FILE = "repeat_tri_lc_causal_layer.log"

OUT_FILE = "loss.png"


def main():
    with open(os.path.join(LOGDIR, LOG_FILE), "r", encoding="utf-8") as f:
        lines = f.readlines()
    losses = [line for line in lines if "loss = " in line]

    values = [part.strip().split("=")[1] for part in losses]
    loss_values = [float(part.strip().split(",")[0]) for part in values]

    steps = [part.strip().split("=")[0] for part in losses]
    loss_steps = [int(part.strip().split("-")[0][4:].strip()) for part in steps]

    # fig, ax = plt.subplots()
    # ax.plot(loss_values)
    plt.plot(loss_steps[1:], loss_values[1:])
    plt.xlabel('iteration')
    plt.ylabel('Training Loss')
    plt.title(TITLE)
    # plt.tight_layout()
    plt.savefig(os.path.join(LOGDIR, OUT_FILE), format='png')

if __name__ == "__main__":
    main()