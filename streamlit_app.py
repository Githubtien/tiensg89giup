# -*- coding: utf-8 -*-
import streamlit as st
from PIL import Image
from PIL import ImageGrab  
from PIL import ImageOps
import random
import os
import cv2 
import numpy as np 
from streamlit_option_menu import option_menu

def Xem_txtmark_hdan(selected):
    st.subheader(":blue["+selected+"]")
    txtmark='''
    **:red[Trước tiên]** dùng điện thoại di động để **:blue[chụp]** Phiếu làm bài trắc nghiệm (gọi tắt là **:blue[PTN]**). 
    Lưu các file ảnh chụp vào trong máy mà trang web này đang hoạt động trên đó. \n
    Đặt tên file ảnh sao cho trong tên đó có chứa kí tự **:blue[PTN_]** (để máy nhận ra đây là PTN, ví dụ là PTN_209_1011.jpg 
    trong đó 209 là mã đề và 1011 là mã số của học viên để tiện xử lí sau nảy). \n
    Khi lấy PTN để làm đáp án thì file ảnh đặt tên dạng PTN_DAP_AN_209.jpg, trong đó 209 là mã đề và kí tự DAP_AN_ 
    để máy nhận ra file đáp án. \n 
    **:red[Sau đó]** làm ba bước theo trên giao diện trang web **:blue[nơi để Upload file]** : \n 
    1. Tải lên file **:blue[PTN đáp án]** để máy căn cứ vào đó mà chấm PTN của học viên. \n 
    2. Tải lên file **:blue[PTN của học viên]** để máy chấm. Khi chấm xong sẽ hiện ra ảnh của PTN đó với kết quả ghi trên ảnh PTN.
    3. Tải xuống file ảnh PTN đã chấm. Tên file sẽ có mã số học viên trong đó. Nếu cần có thể sửa lại tên file. \n 
    **:green[Rồi tiếp tục với từng PTN khác...]**  
    '''
    st.markdown(txtmark)

def Cham_ptn_qua_camera(selected):
    st.subheader(":red["+selected+"]")
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        #st.write('ban vua bat anh nay: ')
        #Xuli_cv2_img_take(cv2_img)
        # Check the type of cv2_img:
        # Should output: <class 'numpy.ndarray'>
        st.image(cv2_img, 'anh cv2 vua bat')

        # Check the shape of cv2_img:
        # Should output shape: (height, width, channels)
        #st.write(cv2_img.shape)    

##########################################################
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


exit()
