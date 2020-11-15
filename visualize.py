import cv2
import matplotlib.pyplot as plt

def visualization(save_fig = False):
    progress = open("detailed_progress.txt", "r").read().split("\n")
    acc_list = list()
    loss_list = list()

    val_acc_list = list()
    val_loss_list = list()
    for line in progress:
        try:
            acc, loss, val_acc, val_loss = line.split(",")
            acc_list.append(float(acc))
            loss_list.append(float(loss))

            val_acc_list.append(float(val_acc))
            val_loss_list.append(float(val_loss))
        except:
            pass
    time_interval = [x for x in range(len(acc_list))]

    axis1 = plt.subplot2grid((2,1), (0,0))
    axis2 = plt.subplot2grid((2,1), (1,0), sharex=axis1)


    axis1.plot(time_interval, acc_list, label="Accuracy")
    axis1.plot(time_interval, val_acc_list, label="Validation Accuracy")
    axis1.legend(loc=2)
    axis2.plot(time_interval,loss_list, label="Loss")
    axis2.plot(time_interval,val_loss_list, label="Validation Loss")
    axis2.legend(loc=2)
    if save_fig:
        plt.savefig("results.png")
    else:
        plt.show()

if __name__ == "__main__":
    visualization(save_fig = False)

