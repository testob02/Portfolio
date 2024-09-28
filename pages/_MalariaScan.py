import streamlit as st
import tensorflow as tf
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

st.set_page_config(layout='wide',page_title="Malaria Scan",page_icon=":mosquito:")
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
    model = tf.keras.models.load_model('Malaria/model.keras')
    return model

model = load_model()

st.title(':blue[MalariaScan :mosquito:]')
st.write(':blue[A model for malaria diagnosis through image classification]')
st.write(':blue[Upload the blood smear images]')

with st.expander(':blue[Options]'):
    uploaded_image = st.file_uploader('Upload Images',accept_multiple_files=True,type=['jpg','png'])

if uploaded_image:
    image_tensor = load_image(uploaded_image[0])
    for img in uploaded_image[1:]:
        image_tensor = tf.concat([image_tensor,load_image(img)],0)
    
    with st.spinner('Scanning blood smears for malaria: where pixels become parasitic detectives'):
        pred = run_inference(model,image_tensor)
        targets = []
        for idx in pred:
            if idx > 0.5:
                targets.append('Uninfected')
            elif idx < 0.5:
                targets.append('Parasitized')
        time.sleep(1)
         
    with st.expander(':blue[Results]', expanded=True):
        st.image(uploaded_image,caption=targets)