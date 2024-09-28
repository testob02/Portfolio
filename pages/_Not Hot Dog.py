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
    pred = model.predict(image_tensor, verbose=0)
    return pred

st.set_page_config(layout='wide',page_title="Not Hot Dog",page_icon=":hotdog:")
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
    model = tf.keras.models.load_model('Not Hot Dog/model.keras')
    return model

model = load_model()

st.title(':blue[Not Hot Dog :hotdog:]')
st.write(':blue[A model to classify images as either Hot Dog or Not]')
st.write(':blue[Inspired by season 4 episode 4 of one of my favorite tech shows, Silicon Valley ]')
st.write(":blue[Watch it if you haven't:laughing:]")
st.link_button(':blue[Watch the video on YouTube]','https://youtu.be/tWwCK95X6go?si=YvSdcw1PRGhizBU7')
st.write(':blue[Take an image using the camera:camera: or upload your image:frame_with_picture:]')

input_option = st.selectbox(label=":blue[Select Input Option]", options=['Upload Image', 'Use Camera'],index=1)

with st.expander(':blue[Options]'):
    if input_option == "Upload Image":
        uploaded_image = st.file_uploader('Upload Image',accept_multiple_files=False,type=['jpg','png'])
    elif input_option == "Use Camera":
        uploaded_image = st.camera_input('Camera')

if uploaded_image:
    image_tensor = load_image(uploaded_image)
    
    with st.spinner('Is that one actually a hot dog?'):
        pred = run_inference(model,image_tensor)
        if pred > 0.5:
            target = 'Not Hot Dog'
        elif pred < 0.5:
            target = 'Hot Dog'
        time.sleep(1)
         
    with st.expander(':blue[Results]', expanded=True):
        st.image(uploaded_image,caption=target)