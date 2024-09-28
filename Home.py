import streamlit as st

st.set_page_config(page_title="My Portfolio App", page_icon=":file_folder:",layout='wide')
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

st.markdown(":blue[I am :orange[**Shittu Teslim**], and this is a showcase of my Machine Learning Projects]")
st.write('---')
st.header(":blue[About Me]")
st.write(":blue[I am just a noob thats passionate about Deep Learning, and exploring the endless things I can do with it]")
st.write('---')
st.header(':blue[App Purpose]')
st.write(":blue[This app serves as a platform to showcase my Machine Learning Projects]")
st.write(":blue[Mostly Computer Vision projects with some NLP here and there]")
st.write(":blue[I would also like to see how much I can actually do before the end of 2024 lol]")
st.write(":blue[No more 'busy with school' excuses :upside_down_face:]")
st.write(":blue[Feel free to explore and check each project on the sidebar]")
st.write(":blue[Each project's page has a link to a github repository]")