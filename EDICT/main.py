from edict_functions import *
import matplotlib.pyplot as plt
def plot_EDICT_outputs(im_tuple):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(im_tuple[0])
    ax1.imshow(im_tuple[1])
    plt.show()

plot_EDICT_outputs(coupled_stablediffusion('A black bear'))