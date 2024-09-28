import streamlit as st
import tensorflow as tf
import numpy as np
import time

#Loading Image Function
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

st.set_page_config(layout='wide',page_title="Dog Diversity",page_icon=":dog:")
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
    model = tf.keras.models.load_model('Dog Breed/model.keras')
    return model

#Get classes
@st.cache_data
def get_classes():
    '''
    Returns:
        labels: A list of class labels with appropriate indexing
    '''
    with open('Dog Breed/labels.txt', 'r') as f:
        labels = [i.split('\n')[0] for i in f.readlines()]
    return labels

classes = get_classes()
model = load_model()

st.title(":blue[Dog Diversity :paw_prints:]")
st.write(":blue[A model that classifies various dog breeds]")
st.write(":blue[Explore the fascinating world of canine diversity]")
st.write(":blue[Upload images:frame_with_picture: or use the camera:camera: in the app and let the paw-some model reveal the breed of your furry friend]")

input_option = st.selectbox(label=":blue[Select Input Option]", options=['Upload Image', 'Use Camera'],index=0)

with st.expander(':blue[Options]',expanded=True):
    if input_option == "Upload Image":
        uploaded_images = st.file_uploader('Upload Image',accept_multiple_files=True,type=['jpg','png'])
    elif input_option == "Use Camera":
        uploaded_images = st.camera_input('Camera')

if uploaded_images:
    if type(uploaded_images) is not list:
        uploaded_images = [uploaded_images]
    image_tensor = load_image(uploaded_images[0])
    for image in uploaded_images[1:]:
        image_tensor = tf.concat([image_tensor,load_image(image)],0)

    with st.spinner(':blue[Pawsing for a moment as the neural network sniffs out details]'):
        pred = run_inference(model, image_tensor)
        targets = []
        for idx in pred:
            targets.append(classes[idx]) 
        time.sleep(1)

    with st.expander(':blue[Results]', expanded=True):
        st.image(uploaded_images,caption=targets)
    
