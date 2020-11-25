
# Only dependencies needed so far
import matplotlib.pyplot as plt
import numpy as np

from .weights import weights_bordermask


def dataset_viewer(dataset, with_weights=False, v_bal=1.5, p_type=True):
    remove_keymap_conflicts({'r', 'l', 'q'})

    if with_weights:
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(13,2.5))
        ax[3].weights, ax[2].sep_mask = precalc_weights(dataset, v_bal=v_bal)
        im = ax[3].imshow(ax[3].weights[0], cmap="jet")
        ax[2].imshow(ax[2].sep_mask[0], cmap='gray')
        if p_type:
            ax[2].title.set_text("Generated mask")
            ax[3].title.set_text("Weight map")
        else:
            ax[2].title.set_text("(c)")
            ax[3].title.set_text("(d)")
        ax[2].axis('off')
        ax[3].axis('off')
        fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,3))

    ax[0].p_type = p_type
    ax[0].dataset = dataset
    ax[0].index = 0
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].imshow(dataset[ax[0].index]['image'].squeeze().numpy(), cmap="gray")
    ax[1].imshow(dataset[ax[0].index]['image'].squeeze().numpy(), cmap="gray")
    ax[1].imshow(dataset[ax[0].index]['mask'].squeeze().numpy(), cmap="nipy_spectral", alpha=0.4)
    if p_type:
        ax[0].title.set_text("Image {}".format(ax[0].index+1))
        ax[1].title.set_text("With Ground truth")
    else:
        ax[0].title.set_text("(a)")
        ax[1].title.set_text("(b)")
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes
    if event.key == 'r':
        previous_slice(fig, ax)
    elif event.key == 'l':
        next_slice(fig, ax)
    elif event.key == 'q':
        ax.close()
    else:
        pass
    fig.canvas.draw()


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def next_slice(fig, ax):
    ax[0].index = (ax[0].index + 1) % len(ax[0].dataset)
    sample = ax[0].dataset[ax[0].index]
    ax[0].images[0].set_array(sample['image'].squeeze().numpy())
    ax[1].images[0].set_array(sample['image'].squeeze().numpy())
    ax[1].images[1].set_array(sample['mask'].squeeze().numpy())
    if ax[0].p_type:
        ax[0].title.set_text("Image {}".format(ax[0].index+1))
        ax[1].title.set_text("With Ground truth")
    else:
        ax[0].title.set_text("(a)")
        ax[1].title.set_text("(b)")
    if len(ax) > 2:
        ax[2].images[0].set_array(ax[2].sep_mask[ax[0].index])
        ax[3].images[0].set_array(ax[3].weights[ax[0].index])
        if ax[0].p_type:
            ax[2].title.set_text("Generated mask")
            ax[3].title.set_text("Weight map")
        else:
            ax[2].title.set_text("(c)")
            ax[3].title.set_text("(d)")


def previous_slice(fig, ax):
    ax[0].index = (ax[0].index - 1) % len(ax[0].dataset)  # wrap around using %
    sample = ax[0].dataset[ax[0].index]
    ax[0].images[0].set_array(sample['image'].squeeze().numpy())
    ax[1].images[0].set_array(sample['image'].squeeze().numpy())
    ax[1].images[1].set_array(sample['mask'].squeeze().numpy())
    if ax[0].p_type:
        ax[0].title.set_text("Image {}".format(ax[0].index+1))
        ax[1].title.set_text("With Ground truth")
    else:
        ax[0].title.set_text("(a)")
        ax[1].title.set_text("(b)")
    if len(ax) > 2:
        ax[2].images[0].set_array(ax[2].sep_mask[ax[0].index])
        ax[3].images[0].set_array(ax[3].weights[ax[0].index])
        if ax[0].p_type:
            ax[2].title.set_text("Generated mask")
            ax[3].title.set_text("Weight map")
        else:
            ax[2].title.set_text("(c)")
            ax[3].title.set_text("(d)")


def precalc_weights(dataset, v_bal=1.5):
    weights, new_masks = [], []
    for x in range(len(dataset)):
        W, new_mask = np.array(weights_bordermask(\
            np.array(dataset[x]['mask'].squeeze().numpy()), v_bal=v_bal))
        weights.append(np.array(W))
        new_masks.append(new_mask > 0)

    return weights, new_masks
