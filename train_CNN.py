import torch
import torch.nn as nn

from CNN import *
from dataset import *
import matplotlib.pyplot as plt



def train_model(model, trainloader, validloader, Note, epochs=100, visualize_learning_curve=True):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    valid_loss_min = np.Inf
    train_losses, test_losses = [], []
    for e in range(epochs):
        model.train()
        running_loss = 0
        tr_accuracy = 0
        for images, labels in trainloader:
            images = images.cuda() #　torch.Size([3000, 1, 48, 48])
            labels = labels.long().cuda() # torch.Size([3000])
            optimizer.zero_grad()
            
            log_ps  = model(images) # torch.Size([3000, 7])
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            tr_accuracy += torch.mean(equals.type(torch.FloatTensor))
        else:
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()
                for images, labels in validloader:
                    images = images.cuda()
                    labels = labels.long().cuda()
                    log_ps = model(images)
                    test_loss += criterion(log_ps, labels)
                    
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(validloader))

            print("Epoch: {}/{} ".format(e+1, epochs),
                "Training Loss: {:.3f} ".format(train_losses[-1]),
                "Training Acc: {:.3f} ".format(tr_accuracy/len(trainloader)),
                "Val Loss: {:.3f} ".format(test_losses[-1]),
                "Val Acc: {:.3f}".format(accuracy/len(validloader)))

            Note.write("Epoch:{}/{}".format(e+1, epochs)+" ")
            Note.write("Training Loss:{:.3f}".format(train_losses[-1])+" ")
            Note.write("Training Acc:{:.3f}".format(tr_accuracy/len(trainloader))+" ")
            Note.write("Val Loss:{:.3f}".format(test_losses[-1])+" ")
            Note.write("Val Acc:{:.3f}".format(accuracy/len(validloader))+"\n")

            if test_loss/len(validloader) <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            test_loss/len(validloader)))
            torch.save(model.state_dict(), 'best_model.pt')
            valid_loss_min = test_loss/len(validloader)

    if visualize_learning_curve:        
        plt.plot(train_losses, 'b', label='Training Loss')
        plt.plot(test_losses, 'r', label='Validation Loss')
        plt.show()
    return model



def main():
    trainloader, validloader = get_dataloaders("./fer2013.csv")
    print('Data Preprocessed and got DataLoaders...')

    with open("note_CNN.text","w+") as Note:
        model = Face_Emotion_CNN()
        if torch.cuda.is_available():
            model.cuda()
            print('GPU Found!!!, Moving Model to CUDA.')
        else:
            print('GPU not found!!, using model with CPU.')

        print('Starting Training loop...\n')
        model = train_model(model, trainloader, validloader, Note, epochs=200)




if __name__ == '__main__':
    main()
    # cd autodl-tmp/Facial-Emotion-Recognition-PyTorch-ONNX-master/PyTorch
    # CUDA_VISIBLE_DEVICES=0 python train_CNN.py