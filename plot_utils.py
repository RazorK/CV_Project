import matplotlib.pyplot as plt


def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, 'r', label='train loss')
    plt.plot(val_loss, 'b', label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss scores')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def plot_accuracy(train, val):
    plt.plot(train, 'r', label='train accuracy')
    plt.plot(val, 'b', label='validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy scores')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def save_accuracy(train, val, file_uri):
    plt.plot(train, 'r', label='train accuracy')
    plt.plot(val, 'b', label='validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy scores')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(file_uri)


def save_loss(train_loss, val_loss, file_uri):
    plt.plot(train_loss, 'r', label='train loss')
    plt.plot(val_loss, 'b', label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss scores')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(file_uri)
