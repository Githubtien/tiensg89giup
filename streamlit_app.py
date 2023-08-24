import cv2
import streamlit as st
import numpy as np
import os
import pickle
from streamlit_option_menu import option_menu
from PIL import Image

def Cham_ptn_qua_camera(selected):
    st.subheader(":blue["+selected+"]")
    dic_dap_an, ch_da = lay_dic_dap_an()
    if ch_da == '':
        st.write(":red[Chưa cung cấp đáp án!]")
    else:    
        st.write(ch_da)

    img_file_buffer = st.camera_input("Chụp Phiếu Trắc Nghiệm",key='CPTNQC')

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


def Cham_ptn_vanhien_da(cv2_img):
    dic_dap_an = {}
    return dic_dap_an

def get_ten_file():
    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    ch=dt_string.replace('/','_')
    ch=ch.replace(' ','_')
    ch=ch.replace(':','')
    return ch + '.jpg'

def Cung_cap_da(selected):
    global dic_dap_an, ch_dap_an
    st.subheader(":red["+selected+"]")
    img_file_buffer = st.camera_input("Chụp Phiếu Trắc Nghiệm Đáp Án",key='CCDA')
    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        #paper = cham_ptn_vanhien(cv2_img,dic_dap_an)
        dic_dap_an = Cham_ptn_vanhien_da(cv2_img)
        if dic_dap_an != {}:
            ch_dap_an = ''
            for keyd in dic_dap_an:
                if  keyd < len(dic_dap_an) - 1 :  
                    ch_dap_an = ch_dap_an + str(keyd+1) + dic_dap_an[keyd] + ', '
                else:    
                    ch_dap_an = ch_dap_an + str(keyd+1) + dic_dap_an[keyd]
            st.write(ch_dap_an)
            chononoffluuda=st.radio("", ('Không lưu (KL)', 'Xac nhan luu (OK)') ,horizontal=True,key='luuda')
            if  'OK' in chononoffluuda:
                tepluub = get_ten_file()
                tepluub = tepluub.replace('.jpg','dap_an_'+made+'.pkl')
                with open(tepluub, 'wb') as fwb:
                    ldap_an=[dic_dap_an,ch_dap_an]
                    pickle.dump(ldap_an, fwb)
                    st.write('Đã lưu đáp án!')    
                    #return ch_dap_an
            else:
                st.write('Cung cấp đáp án chưa thành công!')

###############################

def lay_dic_dap_an():
    dic_dap_an={}
    ch_da = ''
    listd = os.listdir('./dap_an/')
    for tep in listd:
        if 'dap_an_' + '001' + '.pkl' in tep:
            st.write(":red[Chấm theo "+tep[17: -4]+" :]")
            tepluuda = 'dap_an/'+tep
            with open(tepluuda, 'rb') as fwb:
                ldata = pickle.load(fwb)
                dic_dap_an = ldata[0]
                ch_da = ldata[1]
                #st.write(dic_dap_an)
                #st.write(ch_da)
    return dic_dap_an,ch_da

################################################################################################
# main()
st.title("Chấm Điểm Trên Phiếu Trắc Nghiệm Bằng Camera")

with st.sidebar:
    selected = option_menu("Main Menu", ["1. Cung cấp đáp án",   
                                         "2. Chấm PTN qua Camera", "3. Hướng dẫn","4. About" ], default_index=1)
if '1.' in selected:
    ch_dap_an = Cung_cap_da(selected)    
    st.write(ch_dap_an)

elif '2.' in selected:
    Cham_ptn_qua_camera(selected)

elif '3.' in selected:
    Xem_txtmark_hdan(selected)
else:
    st.subheader(":orange[5. About]")
    st.write('App này do tiengs89@gmail.com làm thử năm 2023 để giúp Gv')
###################################

