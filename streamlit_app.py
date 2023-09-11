#https://tiensg89giup-qmcyscahpjmuqyv5jw63jh.streamlit.app/
# reboot app : https://share.streamlit.io/
#lenh chay
#streamlit run streamlit_app.py
#pip install streamlit-camera-input-live
import streamlit as st
import cv2
import numpy as np
import os
import pickle
#from streamlit_option_menu import option_menu
from PIL import Image
from PIL import ImageOps
#from PIL import ImageGrab  
import imutils
from cham_ptn_001_40 import cham_ptn_001_40
from cham_ptn_999_120 import cham_ptn_999_120
from cham_ptn_006_40 import cham_ptn_006_40
from cham_ptn_008_40 import cham_ptn_008_40

from funcs_cham_ptn import brow_img

def get_ten_file_time():
    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    ch=dt_string.replace('/','_')
    ch=ch.replace(' ','_')
    ch=ch.replace(':','')
    return ch

def rut_chda_fromdic(dic_dap_an):
    if dic_dap_an != {}:
        ch_da = ''
        for keyd in dic_dap_an:
            if  keyd < len(dic_dap_an) - 1 :  
                ch_da = ch_da + str(keyd+1) + dic_dap_an[keyd] + ', '
            else:    
                ch_da = ch_da + str(keyd+1) + dic_dap_an[keyd]
        return ch_da

# 1. cho hien tat ca cac mau phieu roi dung selectbox chon lay 1 mau
def Chon_mau_phieu():
    listd = os.listdir('mau_phieu')
    ltep=[]
    for tep in listd:
        #print(tep)
        #if 'PTN_' in tep:
        ltep.append(tep)

    option = st.selectbox(
        ':blue[Chọn một mẫu phiếu mà hv đã làm bài trên PTN theo mẫu đó:]',
        (ltep[i] for i in range(len(ltep))))
    #st.write(option)
    image = Image.open('mau_phieu/'+option)
    #dung ham exif_transpose(imggocinPil) cua modul ImageOps in PIL de xoay lai anh trong st.image
    image = ImageOps.exif_transpose(image)
    new_image = image.resize((600, 400))
    st.image(new_image,channels="BGR")
    mau_phieu_chon=option
    return mau_phieu_chon

    #img = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)
    #im_pil = Image.fromarray(img)

    # For reversing the operation:
    #pil_im_pil = np.asarray(im_pil)

    #dung ham exif_transpose(imggocinPil) cua modul ImageOps in PIL de xoay lai anh trong st.image
    #pil_im_pil = ImageOps.exif_transpose(im_pil)
    #st.image(pil_im_pil, caption='Phiếu trắc nghiệm đã được chấm!')

    #st.write(":red["+ket_qua_thi+"]")
    #return

def Cung_cap_da():
    listd = os.listdir('./dap_an/')
    ltep=[]
    for tep in listd:
        if '_da_' in tep:
            ltep.append(tep[:-4])
    ltep.sort(reverse=True)
    option = st.selectbox(
        ':blue[Chọn một đáp án đã có trước đây?]',
        (ltep[i] for i in range(len(ltep))))

    tepluuda = './dap_an/' + option + '.pkl'
    #st.write(tenfda_dang_dung)

    with open(tepluuda, 'rb') as fwb:
        dic_dap_an = pickle.load(fwb)
        #dic_dap_an = ldata[0]
        ch_da = rut_chda_fromdic(dic_dap_an)
        txtmark ="Đây là đáp án " + ":red["+option+"]" + " bạn đã chọn:"
        st.markdown(txtmark)
        st.write(ch_da)

    if st.checkbox(':blue[Muốn tải lên 1 PTN đáp án khác :]'):
        image_file_da = st.file_uploader(":green[Chọn 1 file ảnh Phiếu Trăc Nghiệm để tải lên.]",type=("png", "jpg"), key='DA')

        if image_file_da is not None:
            pilImg_goc_da = Image.open(image_file_da)
            arrImg = np.array(pilImg_goc_da)
            cv2Img = cv2.cvtColor(arrImg, cv2.COLOR_RGB2BGR)    #mang numpyarray nhung doi sang he mau cua cv2
            cv2Img = cv2.rotate(cv2Img, cv2.ROTATE_90_CLOCKWISE)
            
            #brow_img(cv2Img,'cv2Img')

            #dic_dap_an, ch_da = Cham_ptn_vanhien_da(cv2Img)

            if dic_dap_an != {}:
                ch_da = ''
                for keyd in dic_dap_an:
                    if  keyd < len(dic_dap_an) - 1 :  
                        ch_da = ch_da + str(keyd+1) + dic_dap_an[keyd] + ', '
                    else:    
                        ch_da = ch_da + str(keyd+1) + dic_dap_an[keyd]
                #st.write(ch_dap_an)
                tepluub = "dap_an/" + get_ten_file_time()
                #tepluub = tepluub.replace('.jpg','dap_an_'+made+'.pkl')
                with open(tepluub, 'wb') as fwb:
                    ldap_an=[dic_dap_an,ch_da]
                    pickle.dump(ldap_an, fwb)
                    st.write('Đã lưu đáp án!')    
                    #return ch_da
            else:
                st.write('Cung cấp đáp án mới chưa thành công!')
    return dic_dap_an
    

###############################

def lay_dic_dap_an(dic_dap_an, ch_da,  tenfda_dang_dung):
    #global dic_dap_an, ch_da,tenfda_dang_dung
    dic_dap_an={}
    ch_da = ''
    #listd = os.listdir('./dap_an/')
    tep=tenfda_dang_dung
    st.write(":red[Chấm theo "+tep[17: -4]+" :]")
    tepluuda = 'dap_an/'+tep
    #print(tep)
    st.write(tep)
    st.write(tepluuda)

    with open(tepluuda, 'rb') as fwb:
        ldata = pickle.load(fwb)
        dic_dap_an = ldata[0]
        ch_da = ldata[1]
        #st.write(dic_dap_an)
    st.write(tenfda_dang_dung)
    return dic_dap_an,ch_da,tenfda_dang_dung

################################################################################################
# main()

st.title("Chấm thi auto online trên ảnh của Phiếu Trả Lời Trắc Nghiệm :iphone:")
chonmauphieu=-1
if st.checkbox(':receipt:**:red[Bước 1 : Chọn mẫu phiếu]**'):
    chonmauphieu=1
    mau_phieu_chon = Chon_mau_phieu()
    str_socau = mau_phieu_chon[8:-4]
    st.write('Mẫu phiếu đã chọn là : '+mau_phieu_chon[4:-4])

cungcapdapan=-1
canhbao=-1
if st.checkbox(':banjo:**:red[Bước 2 : Cung cấp đáp án]**') and chonmauphieu == 1:
    cungcapdapan=1
    dic_dap_an = Cung_cap_da()
    if len(dic_dap_an) != int(str_socau):
        canhbao=1
        st.write(':warning:**:orange[Đáp án này có số câu không phù hợp mẫu phiếu chọn chấm! Hãy chọn đáp án khác.]**')    

uploadfile=-1
if st.checkbox(':beginner:**:red[Bước 3 : Upload file image PTN trong máy lên, sau đó auto chấm rồi trả về kết quả.]**') and chonmauphieu == 1 and cungcapdapan==1 and canhbao != 1:
    uploadfile=1
    uploaded_file  = st.file_uploader(":green[Chọn 1 file ảnh Phiếu Trăc Nghiệm để tải lên.]",type=("png", "jpg"), key=4)

    if uploaded_file  is not None:
        #st.write(image_file.name + ' đã được tải lên')
        #st.write(dic_dap_an)
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        if mau_phieu_chon[4:-4]=='001_40':
            paper = cham_ptn_001_40(opencv_image,dic_dap_an)  #chay trong cv2 voi image cv2
            st.image(paper, channels="BGR", caption='Phiếu trắc nghiệm đã được chấm!')

        elif mau_phieu_chon[4:-4]=='999_120':
            paper = cham_ptn_999_120(opencv_image,dic_dap_an)  #chay trong cv2 voi image cv2
            st.image(paper, channels="BGR", caption='Phiếu trắc nghiệm đã được chấm!')
        elif mau_phieu_chon[4:-4]=='006_40':
            paper = cham_ptn_006_40(opencv_image,dic_dap_an)  #chay trong cv2 voi image cv2
            st.image(paper, channels="BGR", caption='Phiếu trắc nghiệm đã được chấm!')
        elif mau_phieu_chon[4:-4]=='008_40':
            paper = cham_ptn_008_40(opencv_image,dic_dap_an)  #chay trong cv2 voi image cv2
            st.image(paper, channels="BGR", caption='Phiếu trắc nghiệm đã được chấm!')
    
    #if '000_120' in lthongtin:
    #    from cham_ptn_000_120 import cham_ptn_000_120
    #    paper, thongbao = cham_ptn_000_120(paper_pre)
    #elif '001_40' in lthongtin:
    #    from cham_ptn_001_40 import cham_ptn_001_40
    #    paper, thongbao = cham_ptn_001_40(paper_pre)
    #elif '002_50' in lthongtin:
    #    from cham_ptn_002_50 import cham_ptn_002_50
    #    paper, thongbao = cham_ptn_002_50(paper_pre)
    #elif '003_50' in lthongtin:
    #    from cham_ptn_003_50 import cham_ptn_003_50
    #    paper, thongbao = cham_ptn_003_50(paper_pre)
    #else:
    #    st.write('PTN không hợp lệ!')    
    #st.image(paper)
    #st.write(thongbao)

    
st.write('---')
if st.checkbox(':iphone:**:green[Phụ lục : Chụp PTN bằng camera online và xử lí auto]**'):
    #Cham_ptn_qua_camera(dic_dap_an)
    #from cham_ptn_000_120 import cham_ptn_000_120
    #image=cv2.imread("PTN_000_120.JPG")
    #paper, thongbao = cham_ptn_000_120(image)
    #st.image(paper)
    st.write('Phần này còn đang thử nghiệm!')
