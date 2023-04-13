import torch
import torch.nn as nn
import clip

from dataset import *
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    trainloader, validloader = get_dataloaders("./fer2013.csv")

    print('Data Preprocessed and got DataLoaders...')

    Note = open('note_CLIP.txt', mode='w')
    # pip install git+https://github.com/openai/CLIP.git
    model, preprocess = clip.load('ViT-B/32', device)
    resize = Resize((224,224))
    # (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

    if torch.cuda.is_available():
        model.cuda()
        print('GPU Found!!!, Moving Model to CUDA.')
    else:
        print('GPU not found!!, using model with CPU.')

    print('Starting Training loop...\n')

    #　image_input = preprocess(Image.open("./test_img/happy.jpeg")).unsqueeze(0).to(device) # ([1, 3, 224, 224])
    #(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    # emotion_dict = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", 'Neutral']
    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in emotion_dict]).to(device)

    accuracy_num = 0
    num =0

    with torch.no_grad():
        for images, labels in validloader:
            images = images #([500, 1, 48, 48])
            labels = labels.long() # ([500])

            for i in range(images.size(0)):
                image_input = images[i].repeat([3,1,1]).unsqueeze(0).to(device) # ([1, 3, 48, 48])
                image_input = resize(image_input)
                emotion_dict = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", 'Neutral']
                text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in emotion_dict]).to(device)

                # image_features = model.encode_image(image_input)
                # text_features = model.encode_text(text_inputs)
                logits_per_image, logits_per_text = model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu() # (1, 7)

                probs = torch.argmax(probs)
                if probs == labels[i] :
                    accuracy_num += 1

                num += 1

                print(accuracy_num/num)
                Note.write("num"+str(num)+": "+str(accuracy_num/num)+"\n")

    print("Val Acc: {:.3f}".format(accuracy_num/num))
    Note.write("Val Acc: {:.3f}".format(accuracy_num/num) + "\n")
    Note.close()


    # Calculate features
    # with torch.no_grad():
    #     image_features = model.encode_image(image_input)
    #     text_features = model.encode_text(text_inputs)
    #     logits_per_image, logits_per_text = model(image_input, text_inputs)
    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()




if __name__ == '__main__':
    main()
    # cd autodl-tmp/Facial-Emotion-Recognition-PyTorch-ONNX-master/PyTorch
    # CUDA_VISIBLE_DEVICES=0 python train_CLIP.py