# -*- coding: utf-8 -*-
import streamlit as st
from PIL import Image
from PIL import ImageGrab  
from PIL import ImageOps
import random
import os
import cv2 
import numpy as np 

from funcs_cham_ptn import *
#from funcs_cham_ptn import Cham_ptn_vanhien
#from funcs_cham_ptn import brow_img

#@st.cache
#def load_image(image_file):
#    img=Image.open(image_file)
#    return img
#color = st.color_picker('Pick A Color', '#00f900')
#st.write('The current color is', color)
#st.write("This is :red[test]")
from streamlit_option_menu import option_menu


st.title("Chấm Điểm Trên Phiếu Trắc Nghiệm với Streamlit")

with st.sidebar:
    selected = option_menu("Main Menu", ["1. Cung cấp đáp án", "2. Upload Phiếu trắc nghiệm cho máy chấm",  
                                         "3. Chấm qua Camera màn hình", "4. Hướng dẫn","5. About" ], default_index=0)
if '1.' in selected:
    ch_dap_an = Cung_cap_da(selected)    
    st.write(ch_dap_an)

elif '2.' in selected:
    Upload_ptn_xulif(selected)    

elif '3.' in selected:
    Cham_ptn_qua_camera(selected)

elif '4.' in selected:
    Xem_txtmark_hdan(selected)
else:
    st.subheader(":orange[5. About]")
    st.write('App này do tiengs89@gmail.com làm thử năm 2023 để giúp Gv')


exit()
st.title("Chấm Điểm Trên Phiếu Trắc Nghiệm")
menu = ["Chấm Điểm Trên Phiếu Trắc Nghiệm","", "1. Cung cấp đáp án", "2. Upload Phiếu trác nghiệm và chấm", "3. Chấm tự động mọi PTN", "4. Chấm bằng Camera màn hình", "About"]
choice = st.sidebar.selectbox("MENU",menu)

if choice == "1. Chấm thi trên Phiếu Trắc Nghiệm":
    st.subheader(":red[1. Chấm thi trên Phiếu Trắc Nghiệm]")
    chononoff=st.radio("", ('OFF : Xem Hướng Dẫn :', 'ON') ,horizontal=True,key=1)
    if chononoff=='ON':
        st.markdown(Xem_txtmark_hdan())
    st.markdown("---")


    st.markdown("Bước 1 : cung cấp đáp án. Có 2 cách: một là nhập trực tiếp, hai là upload ảnh của PTN đáp án lên.")
    #1. Tải lên file **:blue[PTN đáp án]** để máy căn cứ vào đó mà chấm PTN của học viên. \n 
    chononoff2=st.radio("", ('OFF : Nhập trực tiếp đáp án :', 'ON') ,horizontal=True,key=2)

    if chononoff2=='ON':
        otitle = st.text_input('Nhập đáp án vào dòng dưới đây:', '1A, 2B, 3A, 4B ')
        ch=str(otitle)
        sodauphay = ch.count(',')
        st.write('Số đáp án đã nhập khi ENTER là : '+str(sodauphay+1))
        l=ch.split(',')
        #for pt in l:
        #    st.write(pt)
    else:

        st.markdown(''' **:red[Đây là nơi để tải lên tệp ảnh PTN (UPLOAD FILE IMAGE)]** :yellow_heart:''') 
        image_file = st.file_uploader(":green[Chọn 1 tệp ảnh Phiếu Trăc Nghiệm để tải lên.]",type=("png", "jpg"),key=4)
        st.markdown("---")
        
        


        ch_da_bl = ''
        if image_file is not None and 'dap_an_' in image_file.name:
            st.write(':blue[Tệp Ảnh Phiếu trác nghiệm đáp án ( '+image_file.name + ' ) đã được tải lên]')
            
            dic_dap_an = Lay_dap_an_tu_file_anh(image_file,dic_dap_an)
            ch_da_bl, SO_DAU_HOI_IN_DA = Xu_li_dap_an(dic_dap_an)

            if SO_DAU_HOI_IN_DA>0:
                st.markdown(ch_da_bl)
                st.markdown('Hãy làm lại để có được đáp án chuẩn!')
            else:
                st.markdown(ch_da_bl)
                st.markdown('TT')

        st.markdown("---")

        if SO_DAU_HOI_IN_DA > 0 and image_file is not None and 'dap_an_' not in image_file.name:
            st.markdown('khong xl f nay vi da xau')
        if SO_DAU_HOI_IN_DA == 0 and image_file is not None:
            
            st.markdown(''' **:red[Đây là nơi để tải lên tệp ảnh PTN (UPLOAD FILE IMAGE)]** :yellow_heart:''') 
            image_file2 = st.file_uploader(":green[Cbbbbbb bbbbb họn 1 tệp ảnh Phiếu Trăc Nghiệm để tải lên.]",type=("png", "jpg"), key=2)
            st.markdown('ok da t')
            
            #st.write(image_file.name + ' đã được tải lên')
            #dic_dap_an,SO_DAU_HOI_IN_DA = Lay_dap_an_tu_file_anh(image_file,dic_dap_an)
            #if SO_DAU_HOI_IN_DA > 0:
            #    dic_dap_an,SO_DAU_HOI_IN_DA,ch_da_bl = Xem_gon_dap_an(dic_dap_an,SO_DAU_HOI_IN_DA,ch_da_bl)
            #    st.write(ch_da_bl)
            #    st.write(":red[Bước 2 :] "+ " :blue[Bây giờ hãy chọn một file ảnh Phiếu Trăc Nghiệm dap an khac.]")
            #else:
            #    dic_dap_an,SO_DAU_HOI_IN_DA,ch_da_bl = Xem_gon_dap_an(dic_dap_an,SO_DAU_HOI_IN_DA,ch_da_bl)
            #    st.write(ch_da_bl)

            #    st.write(":red[Bước 2 :] "+ " :blue[Bây giờ hãy chọn một file ảnh Phiếu Trăc Nghiệm của học viên để tải lên chấm.]")

            #    pilImg_goc2 = Image.open(image_file)
            #    arrImg2 = np.array(pilImg_goc2)
            #    cv2Img2 = cv2.cvtColor(arrImg2, cv2.COLOR_RGB2BGR)    #mang numpyarray nhung doi sang he mau cua cv2
            #    cv2Img2 = cv2.rotate(cv2Img2, cv2.ROTATE_90_CLOCKWISE)

            #    paper2, ket_qua_thi = Cham_ptn_vanhien_hv(cv2Img2, dic_dap_an)

            #    st.write("Cham xong. Duoi day la anh PTN da cham")

            #    img2 = cv2.cvtColor(paper2, cv2.COLOR_BGR2RGB)
            #    im_pil2 = Image.fromarray(img2)

                # For reversing the operation:
            #    pil_im_pil2 = np.asarray(im_pil2)

                #dung ham exif_transpose(imggocinPil) cua modul ImageOps in PIL de xoay lai anh trong st.image
                #pil_im_pil2 = ImageOps.exif_transpose(im_pil2)

            #    st.image(pil_im_pil2, caption='Phiếu trắc nghiệm đã được chấm!')

            #    #st.write(":red["+ket_qua_thi+"]")
         

else:
    st.subheader(":blue[2. About]")
    st.write('App này do tiengs89@gmail.com làm thử năm 2023 để giúp Gs Đạo')

    
