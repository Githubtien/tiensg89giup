# -*- coding: utf-8 -*-
import streamlit as st
from PIL import Image
from PIL import ImageGrab  
from PIL import ImageOps
from PIL import ImageFilter
import numpy as np 
#import cv2 
import random
import os
from streamlit_option_menu import option_menu
#from funcs_cham_ptn import *

def Cham_ptn_qua_camera(selected):
    st.subheader(":red["+selected+"]")
    img_file_buffer = st.camera_input("Take a picture",key='cmrtien')

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        #cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        #st.write('ban vua bat anh nay: ')
        #Xuli_cv2_img_take(cv2_img)
        # Check the type of cv2_img:
        # Should output: <class 'numpy.ndarray'>
        #st.image(bytes_data, 'anh cv2 vua bat')
        # in order to deal with '\' in paths)

 
        # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
        st.image(bytes_data, 'anh cv2 vua bat')

        #image  = image.rotate(-90)
 
        # Saving the Image Under the name Edge_Sample.png


        # Check the shape of cv2_img:
        # Should output shape: (height, width, channels)
        #st.write(cv2_img.shape)    



###########################################################
st.title("Chấm Điểm Trên Phiếu Trắc Nghiệm với Streamlit")

with st.sidebar:
    selected = option_menu("Main Menu", ["1. Cung cấp đáp án", "2. Upload Phiếu trắc nghiệm cho máy chấm",  
                                         "3. Chấm qua Camera màn hình", "4. Hướng dẫn","5. About" ], default_index=0)

if '3.' in selected:
    Cham_ptn_qua_camera(selected)

else:
    st.subheader(":orange[5. About]")
    st.write('App này do tiengs89@gmail.com làm thử năm 2023 để giúp Gv')


