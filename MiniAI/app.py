
from fastai.vision.all import *
import gradio as gr
import torch; 
def label_func(f): 
    return f[0].isupper()



def main():
    print("running")
    print(torch.cuda.is_available())
    path = untar_data(URLs.PETS)

    print(path.ls())

    files = get_image_files(path/"images")
    print(len(files))



    dls = ImageDataLoaders.from_name_func(
    path, 
    files, label_func, 
    item_tfms=Resize(224),
    batch_size=4  )

    dls.show_batch()

    learn = vision_learner(dls, efficientnet_b0, metrics=error_rate)
    learn.fine_tune(1)
    learn.export('modell.pkl')



    print (learn.predict(files[0]))
    
    learn.path = Path('.')
    learn.export()

    


if __name__ == '__main__':
    main()
