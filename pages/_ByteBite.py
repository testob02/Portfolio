import streamlit as st
import tensorflow as tf
import numpy as np
import time

def load_image(image):
    '''
    Arguments:
        image: UploadedFile type from streamlit file input widgets
    Returns:
        img: A tensor of shape (1,224,224,3) representing input image
    '''
    img = tf.image.decode_image(image.read(), channels=3)
    img = tf.image.resize(img,[224,224])
    img = tf.expand_dims(img,0)
    return img

#Function to get predictions
def run_inference(model,image_tensor):
    '''
    Arguments:
        model: keras model used to run inference
        image_tensor: a tensor the model runs inference on
    Returns:
        pred: an array of predictions with class index
    '''
    pred = np.argmax(model.predict(image_tensor, verbose=0),-1)
    return pred

st.set_page_config(layout='wide',page_title="ByteBite",page_icon=":hamburger:")
st.markdown(
"""
<style> 
.stDeployButton {
    visibility: hidden;
}
#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

#Load Model
@st.cache_resource
def load_model():
    '''
    Returns:
        model: a keras model
    '''
    model = tf.keras.models.load_model('Food/model2.keras')
    return model

#Get classes
@st.cache_data
def get_classes():
    '''
    Returns:
        labels: A list of class labels with appropriate indexing
    '''
    with open('Food/labels.txt', 'r') as f:
        labels = [i.split('\n')[0] for i in f.readlines()]
    return labels

classes = get_classes()
model = load_model()

st.title(":blue[ByteBite :pizza:]")
st.write(":blue[Upload your images and ByteBite will classify your food faster than you can say byte me]")

input_option = st.selectbox(':blue[Select your input option]',options=['Upload Image','Use Camera'], index=1)

with st.expander(':blue[Options]', expanded=True):
    if input_option == "Upload Image":
        uploaded_images = st.file_uploader(':blue[Upload Image]',accept_multiple_files=True,type=['jpg','png'])
    elif input_option == "Use Camera":
        uploaded_images = st.camera_input(':blue[Camera]')

if uploaded_images:
    if type(uploaded_images) is not list:
        uploaded_images = [uploaded_images]
    image_tensor = load_image(uploaded_images[0])
    for image in uploaded_images[1:]:
        image_tensor = tf.concat([image_tensor,load_image(image)],0)

    with st.spinner(':blue[ByteBite is busy crunching and slicing through pixels to serve you a hot result]'):
        pred = run_inference(model, image_tensor)
        targets = []
        for idx in pred:
            targets.append(classes[idx]) 
        time.sleep(1)
    
    with st.expander(':blue[Results]', expanded=True):
        st.image(uploaded_images,caption=targets)
