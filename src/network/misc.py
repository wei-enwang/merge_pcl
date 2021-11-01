import matplotlib.pyplot as plt

def plot_loss(epochs, train_loss, title="", filename=None):

    # Plot epoch loss
    plt.figure(facecolor="white")
    plt.plot(range(epochs), train_loss, label='training')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()