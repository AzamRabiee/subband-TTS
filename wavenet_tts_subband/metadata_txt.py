import os

METADATA_MASTER = "/media/cnsl/Datasets/LJSpeech-1.1/trimmed/wavenet_tts/train.txt"
OUTPUT_PATH = "/media/cnsl/Datasets/LJSpeech-1.1/sub_datasets_redundant/train"
N_LEVELS = 8

def main():
    # load wavenet_tts_master metadata text file
    with open(METADATA_MASTER, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i in range(N_LEVELS):
        # open di_train_tts.txt for writing metadata info for training subbands
        out_file = os.path.join(OUTPUT_PATH, "d%d/d%d_train_tts.txt" % (i+1, i+1))
        with open(out_file, 'w', encoding='utf-8') as f:
            for line in lines:
                parts = line.strip().split("|")
                parts[0] = "d%d_%s.npy" % (i+1, parts[0][:-10])
                parts[1] = "../../../trimmed/wavenet_tts/%s" % (parts[1])
                f.write('|'.join([str(x) for x in parts]) + '\n')


if __name__ == "__main__":
    main()