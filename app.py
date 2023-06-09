import cv2
import numpy as np
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

# @st.cache_data
def load_image():
    if 'use_sample_image' not in st.session_state:
        st.session_state['use_sample_image'] = False

    if st.session_state['use_sample_image']:
        image_file = open('sample.jpg', 'rb')
        st.write('Done playing with the sample image?')
        if st.button('Upload your own'):
            st.session_state['use_sample_image'] = False
            st.experimental_rerun()
    else:
        image_file = st.file_uploader("Upload Images")
        st.write('Or, try out the sample image:')
        if st.button('Use sample image'):
            st.session_state['use_sample_image'] = True
            st.experimental_rerun()
        if image_file:
            cropped = st_cropper(Image.open(image_file), realtime_update=True, box_color='#61C761', aspect_ratio=(1,1))

    st.write('Colony Counter is a tool that allows you to count bacteria colonies in a petri dish. It is recommended that your bacteria colonies are glowing, as I have not tested Colony Counter with tan colonies. I also suggest using a high contrast image, free of glare and shadows. The image can be any resolution, however, Colony Counter will run faster (and is more customizable) with a smaller image. The default settings are optimized for a 1024x1024 image, so please consider resizing it before uploading. Finally, this is a work in progress, so please send feedback to sigkukla@gmail.com.')

    # If no image is uploaded, stop the app
    if image_file is None:
        # st.error('No image uploaded', icon='❌')
        st.stop()

    if st.session_state['use_sample_image']:
        image = cv2.imread('sample.jpg')
    else:
        image = np.array(cropped)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if image.shape[0] > 1024 or image.shape[1] > 1024:
        st.warning('Image is very large, resizing to 1024x1024')
        image = cv2.resize(image, (1024, 1024))

    # Contrast adjustment option
    alpha = st.slider('Contrast', 0.0, 2.0, 1.0, 0.1)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    
    # Get grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, gray

def detect(image, gray):
    # threshold to get binary image
    threshold_slider = st.slider('Threshold', 0, 255, (120,255))
    ret,thresh = cv2.threshold(gray,threshold_slider[0],threshold_slider[1],0)

    # perform an opening operation to remove small chunks
    kernel_size = st.slider('Kernel size', 0, 10, 3)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((kernel_size,kernel_size),np.uint8))

    # Finding contours
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    area_slider = st.slider('Min area', 0, 3000, (20,1000))

    contours = [contour for contour in contours if cv2.contourArea(contour) >= area_slider[0] and cv2.contourArea(contour) <= area_slider[1]]
    num_colonies = len(contours)
    st.header(f'{num_colonies} colonies detected')
    # print("Number of Contours found = " + str(len(contours)))
    cv2.putText(image, f"Number of colonies found = {num_colonies}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
    # Draw all contours by using -1 parameter
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    
    # Put the processed image in the streamlit app
    stimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # streamlit needs RGB
    st.image(stimage, use_column_width=True)

    # Add a download button for the processed image
    file = cv2.imencode('.jpg', image)[1].tobytes() # cv2 converts to RGB in encode automatically
    st.download_button(label='Download image', data=file, file_name=f'{num_colonies}_colonies.jpg', mime='image/jpg')

if __name__ == '__main__':
    st.set_page_config(page_title='Colony Counter', page_icon='🦠', layout='centered', initial_sidebar_state='auto', menu_items={
        'About': """# Colony Counter

        Created and maintained by Sigmond Kukla 
        https://sigmondkukla.dev
        https://github.com/PicoPlanetDev
        Send bugs and feedback to sigkukla@gmail.com"""
    })

    # Add a title to the app
    st.title('🧫 Colony Counter')
    
    # Prompt to get image
    info = st.info('Please select an image', icon='📷')
    image, gray = load_image()

    # Remove the select image prompt
    info.info('Processing image...', icon='⚙️')
    detect(image, gray)

    # Assume the image is processed, put a done message
    info.info('Done, scroll down to see results!', icon='✅')