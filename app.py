import cv2
import numpy as np
import streamlit as st

# @st.cache_data
def load_image():
    image_file = st.file_uploader("Upload Images")
    
    # If no image is uploaded, stop the app
    if image_file is None:
        st.error('No image uploaded', icon='❌')
        st.stop()    

    # Get the image as a numpy array
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), -1)

    # Contrast adjustment option
    alpha = st.slider('Contrast', 0.0, 3.0, 1.0, 0.1)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    
    # Get grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, gray

def detect(image, gray):
    # threshold to get binary image
    thresh_min = st.slider('Threshold min', 0, 255, 120)
    thresh_max = st.slider('Threshold max', 0, 255, 255)
    ret,thresh = cv2.threshold(gray,thresh_min,thresh_max,0)

    # perform an opening operation to remove small chunks
    kernel_size = st.slider('Kernel size', 0, 20, 3)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((kernel_size,kernel_size),np.uint8))

    # Finding contours
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    min_area = st.slider('Min area', 0, 100, 1)
    max_area = st.slider('Max area', 0, 2000, 1000)
    
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area and cv2.contourArea(contour) <= max_area]
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
    # Add a title to the app
    st.title('Colony Counter')
    
    # Prompt to get image
    info = st.info('Please select an image', icon='📷')
    image, gray = load_image()

    # Remove the select image prompt
    info.info('Processing image...', icon='⚙️')
    detect(image, gray)

    # Assume the image is processed, put a done message
    info.info('Done!', icon='✅')