from fastai.vision.all import *
import gradio as gr
import os


def is_cat(f): 
    return f[0].isupper()


learn = load_learner('model.pkl')


categories = ['Dog', 'Cat']


def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


image = gr.Image()
label = gr.Label()


script_dir = os.path.dirname(os.path.abspath(__file__))
example = os.path.join(script_dir, "Etku.jpg")


if not os.path.exists(example):
    print("Error: Image file 'Etku.jpg' not found in the same directory.")
    exit(1)


interface = gr.Interface(
    fn=predict, 
    inputs =image, 
    outputs = label, 
    examples = [[example]],
    title = "Pet Classifier",
    description = "This model predicts whether an image contains a dog or a cat.",
    theme='soft',
    allow_flagging= 'never',
    live = true,
   )


interface.launch(share=False)
