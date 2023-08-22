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
    Cham_ptn_qua_camera(selected)

elif '4.' in selected:
    Xem_txtmark_hdan(selected)
else:
    st.subheader(":orange[5. About]")
    st.write('App này do tiengs89@gmail.com làm thử năm 2023 để giúp Gv')


