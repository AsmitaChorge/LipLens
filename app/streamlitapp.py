# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

#Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        # Prepare video for display while maintaining model input format
        def prepare_video_for_display(video):
            display_video = []
            video = video.numpy()  # Convert tensor to numpy array
            
            for frame in video:
                frame = np.squeeze(frame)  # Remove extra dimensions if present
                
                # Reverse normalization and scale to 0-255
                frame = (frame * 127.5 + 127.5).clip(0, 255)
                
                # Convert to uint8
                frame = frame.astype(np.uint8)

                # If grayscale, convert to RGB for display
                if frame.ndim == 2:  # shape: (H, W)
                    frame = np.stack([frame] * 3, axis=-1)  # shape: (H, W, 3)

                display_video.append(frame)

            return display_video

        # Save and display the video (converted to RGB)
        display_video = prepare_video_for_display(video)
        imageio.mimsave('animation.gif', display_video, fps=10)
        st.image('animation.gif', width=400) 

        # Keep original video tensor for model prediction
        video = tf.cast(video, tf.float32)  # Ensure proper dtype
        

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

