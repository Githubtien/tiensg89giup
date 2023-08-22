# -*- coding: utf-8 -*-
import streamlit as st
from PIL import Image
from PIL import ImageGrab  
from PIL import ImageOps
import random
import os
import numpy as np 

from funcs_cham_ptn import *
from streamlit_option_menu import option_menu


st.title("Chấm Điểm Trên Phiếu Trắc Nghiệm với Streamlit")

with st.sidebar:
    selected = option_menu("Main Menu", ["1. Cung cấp đáp án", "2. Upload Phiếu trắc nghiệm cho máy chấm",  
                                         "3. Chấm qua Camera màn hình", "4. Hướng dẫn","5. About" ], default_index=0)

if '3.' in selected:
    st.subheader(":red["+selected+"]")
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()

else:
    st.subheader(":orange[5. About]")
    st.write('App này do tiengs89@gmail.com làm thử năm 2023 để giúp Gv')


