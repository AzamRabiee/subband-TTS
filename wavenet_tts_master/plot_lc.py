import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def image_save(image, path, info=None):
    image = np.squeeze(image.transpose())
    fig, ax = plt.subplots()
    im = ax.imshow(
        image,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    plt.savefig(path, format='png')
