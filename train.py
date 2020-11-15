import argparse
from utils import Create_Dataset
from model import *
import numpy as np
import random
from tqdm import tqdm

class Training:
    def __init__(self,args):
        self.class_folder, self.lr, self.batch_size, self.resume, self.img_size, self.epochs, self.device, self.new_data = \
            args.class_folder, args.lr, args.batch_size, args.resume, args.img_size, args.epochs, args.device, args.new_data
        self.dataset, self.class_file = args.dataset, args.class_file
        self.loss_function = None  # will contain loss_function
        self.net = None #Will contain model
    def train(self):

        create_dataset = Create_Dataset(im_size = self.img_size, dataset = self.dataset,
                                        c_folder = self.class_folder, class_file = self.class_file)

        if self.new_data:
            create_dataset.prepare_data()
        print("Classes:\n\t", create_dataset.Classes)

        if self.device == "":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print("Using GPU...")
            else:
                self.device = torch.device("cpu")
                print("Using CPU...")

        self.net = Net(len(create_dataset.Classes), img_size = self.img_size).to(self.device)

        if self.resume:
            progress_file = open("epoch_progress.txt","r")
            lines = progress_file.read().split("\n")
            accuracies = [line[0].split(",")[0] for line in lines]
            progress_file.close()
            best_acc = np.max(accuracies)
            self.net.load_state_dict(torch.load("models/last.pth.tar"))
        else:
            best_acc = 0.0
            f_log = open("detailed_progress.txt","w")
            f_log.close()
            with open("epoch_progress.txt","w") as f:
                f.write("Epoch\t\tAccuracy\tLoss\t        Val_Accuracy\tVal_loss\n")

        dataset = np.load(self.dataset, allow_pickle=True)
        print("Loading Dataset...(It may take a while depending on your dataset size)")
        x = torch.Tensor([data[0] for data in dataset])# Inputs
        x = x / 255.0

        y = torch.Tensor([data[1] for data in dataset]) #Expected outputs

        val_perc = 0.2 # Validation Percentage
        val_size = int(len(x) * val_perc)

        train_x = x[:-val_size]
        train_y = y[:-val_size]

        self.test_x = x[-val_size:]
        self.test_y = y[-val_size:]
        print("Training Data Length: ", len(train_x), "\nTest Data Length: ", len(self.test_x), "\n")


        optimizer = optim.Adam(self.net.parameters(), lr = self.lr)
        self.loss_function = nn.MSELoss()
        acc_list = list()
        loss_list = list()
        with open("detailed_progress.txt", "a") as f_log:
            for epoch in range(self.epochs):
                for i in tqdm(range(0, len(train_x), self.batch_size)):
                    batch_x = train_x[i:i + self.batch_size].view(-1,1, self.img_size, self.img_size)
                    batch_y = train_y[i:i + self.batch_size]
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    self.net.zero_grad()
                    output = self.net(batch_x)
                    match = [torch.argmax(k) == torch.argmax(h) for k, h in zip(output, batch_y)]

                    acc = match.count(True) / len(match)
                    acc_list.append(acc)

                    loss = self.loss_function(output, batch_y)
                    loss_list.append(loss)

                    loss.backward()
                    optimizer.step()

                    if i % 320 == 0:
                        val_acc, val_loss = self.mini_test()
                        f_log.write(f"{round(float(acc), 2)},"
                                    f"{round(float(loss), 4)},{round(float(val_acc), 2)},{round(float(val_loss), 4)}\n")
                acc, loss = sum(acc_list) / len(acc_list), sum(loss_list) / len(loss_list) # getting average train acc and loss
                acc_list, loss_list = [], []
                val_acc, val_loss = self.test()# getting average test acc and loss
                if val_acc > best_acc:
                    torch.save(self.net.state_dict(), "models/best.pth.tar")

                torch.save(self.net.state_dict(), "models/last.pth.tar")
                if epoch % 3 == 0:
                    torch.save(self.net.state_dict(),f"models/backup{epoch}.pth.tar")
                print(f"Accuracy: {round(float(acc), 4)}  Loss: {round(float(loss), 4)}")
                print(f"Validation Accuracy: {round(float(val_acc), 4)}   Validation Loss: {round(float(val_loss), 4)}")

                with open("epoch_progress.txt","a") as f:
                    f.write(f"{epoch}\t\t{round(float(acc), 4)}\t\t{round(float(loss), 4)}\t\t{round(float(val_acc), 4)}\t\t{round(float(val_loss), 4)}\n")


    def test(self):
        val_acc_list = list()
        val_loss_list = list()
        with torch.no_grad():
            for l in range(0, len(self.test_x), self.batch_size):
                batch_x = self.test_x[l:l + self.batch_size].view(-1, 1, self.img_size, self.img_size)
                batch_y = self.test_y[l: l + self.batch_size]
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                output = self.net(batch_x)
                match = [torch.argmax(n) == torch.argmax(m) for n, m in zip(output, batch_y)]
                acc = match.count(True) / len(match)
                val_acc_list.append(acc)

                loss = self.loss_function(output, batch_y)
                val_loss_list.append(loss)

        return sum(val_acc_list) / len(val_acc_list), sum(val_loss_list) / len(val_loss_list)

    def mini_test(self):
        rand_batch = random.randint(0,len(self.test_x)-self.batch_size)# got random num to test random images
        batch_x = self.test_x[rand_batch : rand_batch + self.batch_size].view(-1, 1, self.img_size, self.img_size)
        batch_y = self.test_y[rand_batch : rand_batch + self.batch_size]
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

        output = self.net(batch_x)
        match = [torch.argmax(n) == torch.argmax(m) for n, m in zip(output, batch_y)]

        acc = match.count(True) / len(match)
        loss = self.loss_function(output, batch_y)

        return acc, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_folder", type = str, default ="data/images",help = "Put your class folders into data/images")
    parser.add_argument("--class_file", type = str, default = "data/classes.txt", help = "File that contains your classes")
    parser.add_argument("--dataset", type = str, default = "data/dataset.npy", help = "Dataset name that will be created")
    parser.add_argument("--lr", type = float, default = 1e-3, help="Learning Rate")
    parser.add_argument("--batch_size", type= int , default = 32, help = "Total batch Size")
    parser.add_argument("--resume", action='store_true', help = "Continue from last.pth.tar")
    parser.add_argument("--img_size", type= int, default = 64, help = "Image sizes of input images")
    parser.add_argument("--epochs", type = int , default = 100, help = "How many times the model will be trained")
    parser.add_argument("--device", type = str, default = "", help = "device_id")
    parser.add_argument("--new_data", action = "store_true", help = "If it is your first time with input data")
    args = parser.parse_args()
    print(args)

    model_training = Training(args)
    model_training.train()