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


st.set_page_config(layout='wide',page_title="Animal Kingdom",page_icon=":lion_face:")
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
    model = tf.keras.models.load_model('Animal/model.keras')
    return model

#Get classes
@st.cache_data
def get_classes():
    '''
    Returns:
        labels: A list of class labels with appropriate indexing
    '''
    with open('Animal/labels.txt', 'r') as f:
        labels = [i.split('\n')[0] for i in f.readlines()]
    return labels

classes = get_classes()
model = load_model()

st.title(':blue[Animal Kingdom :lion_face:]')
st.write(":blue[Explore the vast diversity of the animal kingdom with this image classifier]")
st.write(":blue[From majestic lions to tiny insects, Animal Kingdom accurately classifies creatures]")
st.write(':blue[Take an image or Upload an image]')

input_option = st.selectbox(label=":blue[Select Input Option]", options=['Upload Image', 'Use Camera'],index=1)

with st.expander(':blue[Options]'):
    if input_option == "Upload Image":
        uploaded_image = st.file_uploader('Upload Image',accept_multiple_files=False,type=['jpg','png'])
    elif input_option == "Use Camera":
        uploaded_image = st.camera_input('Camera')

if uploaded_image:
    image_tensor = load_image(uploaded_image)

    with st.spinner(':blue[Animal Kingdom is hard at work, classifying the creature in your image :bird:]'):
        pred = run_inference(model, image_tensor)
        target = classes[pred[0]]
        time.sleep(1)

    with st.expander(':blue[Results]', expanded=True):
        st.image(uploaded_image,caption=target)