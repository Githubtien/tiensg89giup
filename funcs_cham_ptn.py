import streamlit as st
from PIL import Image
from PIL import ImageGrab  
from PIL import ImageOps
import random
import os
import opencv-python as cv2 
import numpy as np 
from imutils.perspective import four_point_transform
import imutils
import pickle
global SO_DAU_HOI_IN_DA, dic_dap_an, ch_dap_an
SO_DAU_HOI_IN_DA = 0
dic_dap_an = {}
ch_dap_an=''

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


def Check_YN_khoivuong(cv2_img_xoay):
    gray = cv2.cvtColor(cv2_img_xoay, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)

    cv2.drawContours(cv2_img_xoay, cnts, -1, (0,0,255), 4)

def Xuli_cv2_img_take(cv2_img):
    cv2_img_xoay = cv2.rotate(cv2_img, cv2.ROTATE_90_CLOCKWISE)
    Check_YN_khoivuong(cv2_img_xoay)
    st.image(cv2_img_xoay, caption='KQ')
    #brow_img(cv2_img_xoay,'Toi thu')

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

def Cung_cap_da(selected):
    global dic_dap_an, ch_dap_an
    st.subheader(":red["+selected+"]")
    #chononoff1=st.radio("Có hai cách, chọn một cách ", ('C1: nhập trực tiếp', 'C2: tải ảnh PTN đáp án lên'),index=0 ,horizontal=True,key=1)
    #if 'C2' in chononoff1:
    image_file_da = st.file_uploader(":green[Chọn 1 tệp ảnh Phiếu Trăc Nghiệm đáp án để tải lên.]",type=("png", "jpg"),key=1)
        
    #ch_da_bl = ''
    if image_file_da is not None and 'dap_an_' in image_file_da.name:
        st.write(':blue[Tệp Ảnh Phiếu trác nghiệm đáp án ( '+image_file_da.name + ' ) đã được tải lên]')
        pilImg_goc = Image.open(image_file_da)
        arrImg = np.array(pilImg_goc)
        cv2Img = cv2.cvtColor(arrImg, cv2.COLOR_RGB2BGR)    #mang numpyarray nhung doi sang he mau cua cv2
        cv2Img = cv2.rotate(cv2Img, cv2.ROTATE_90_CLOCKWISE)
        dic_dap_an = Cham_ptn_vanhien_da(cv2Img, dic_dap_an)
        #st.write(dic_dap_an)
        #doi sang dang chuoi
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
            tepluub = tepluub.replace('.jpg','dap_an_.pkl')
            with open(tepluub, 'wb') as fwb:
                ldap_an=[dic_dap_an,ch_dap_an]
                pickle.dump(ldap_an, fwb)
                st.write('Đã lưu đáp án!')    
                #return ch_dap_an

    #else:
    #    exit(st.write("Không lấy được đáp án từ file này!"))

    
def Upload_ptn_xulif(selected):
    global dic_dap_an, ch_dap_an
    st.subheader(":blue["+selected+"]")
    st.write("với đáp án này : ")
    st.write(ch_dap_an)
    image_file = st.file_uploader(":green[Chọn 1 tệp ảnh Phiếu Trăc Nghiệm để tải lên.]",type=("png", "jpg"),accept_multiple_files=False,key=4)



def brow_img(image,namewin):
    cv2.namedWindow(namewin, cv2.WINDOW_NORMAL) 
    cv2.imshow(namewin, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return


def save_dap_an(ma_de, dic_dap_an):
    tepluub = 'dap_an_md_' + ma_de + '.pkl'
    with open(tepluub, 'wb') as fwb:
        pickle.dump(dic_dap_an, fwb)
    print('Da luu dic dap an vao file:', tepluub)
    return
        
def lay_dap_an(ma_de):
    tepluuda = 'dap_an_' + ma_de + '.pkl'
    with open(tepluuda, 'rb') as fwb:
        dic_dap_an = pickle.load(fwb)
    return dic_dap_an

def lay_dap_an_ftxt_save(ma_de):
    filebai  = "dap_an_"+ma_de+".txt"
    with open(filebai, mode='r') as ff:
        textall = ff.read()
    lbai = list(textall.split('\n'))
    lbain=[]
    for lc in lbai:
        if not lc == '':
            lbain.append(lc)
    dic1={}
    i=-1
    for lc in lbain:
        i=i+1
        x=dic1.items()
        dic1[i]=lc[-1]
    #print(dic1)
    dic_dap_an = dic1
    save_dap_an(ma_de, dic_dap_an)

    dic_dap_an = lay_dap_an(ma_de)
    return dic_dap_an
####
def get_x_ver0(s):
   s = cv2.boundingRect(s)
   return s[0]

def get_y_ver1(s):
   s = cv2.boundingRect(s)
   return s[1]

def Find_4kv_Black_Big_Top_Bot(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
    #brow_img(thresh,'XXXXXXXXXXX')
    cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)
    
    cnts = sorted(cnts, key = get_y_ver1)

    cnts_lay=[]
    whmin = 30
    whmax = 100
    for c in cnts[:40]:
        x,y,w,h = cv2.boundingRect(c)
        if  (whmin <= w <= whmax) and (whmin <= h <= whmax) : #and len(approx)==4:
            cnts_lay.append(c)
            #print(w,h)
            #cv2.drawContours(image, [c], 0, (0,0,255), 4)
            #brow_img(image,'image')
            if len(cnts_lay)==2:
                break
    if len(cnts_lay)<2:
        exit('PTN khong dat!')        
    cnts = sorted(cnts, key = get_y_ver1, reverse=True)
    for c in cnts[:40]:
        x,y,w,h = cv2.boundingRect(c)
        if  (whmin <= w <= whmax) and (whmin <= h <= whmax) : #and len(approx)==4:
            cnts_lay.append(c)
            #print(w,h)
            #cv2.drawContours(image, [c], 0, (0,0,255), 4)
            #brow_img(image,'image')
            if len(cnts_lay)==4:
                break
    if len(cnts_lay)<4:
        exit('PTN khong dat!')        
    cnts_lay = sorted(cnts_lay,key=get_y_ver1)
    cnt_4c_TOP = cnts_lay[:2]
    cnt_4c_BOT = cnts_lay[-2:] 
    cnt_4c_TOP = sorted(cnt_4c_TOP, key= get_x_ver0)
    cnt_4c_BOT = sorted(cnt_4c_BOT, key= get_x_ver0)
    #paper = Scanhoa_from_4dinh_of4kv_mark(image,cnt_4c_TOP, cnt_4c_BOT)
    return cnt_4c_TOP, cnt_4c_BOT

def Scanhoa_from_4dinh_of4kv_mark(img,cnts_4KV_top, cnts_4KV_bot):
    (x1,y1,w1,h1)=cv2.boundingRect(cnts_4KV_top[0])
    (x2,y2,w2,h2)=cv2.boundingRect(cnts_4KV_top[-1])
    (x3,y3,w3,h3)=cv2.boundingRect(cnts_4KV_bot[0])
    (x4,y4,w4,h4)=cv2.boundingRect(cnts_4KV_bot[-1])
    X1,Y1 = x1, y1
    X2,Y2 = x2+w2, y2
    X3,Y3 = x3,y3+h3
    X4,Y4 = x4+w4,y4+h4
    pts = np.array(eval("[(X1, Y1), (X3,Y3),(X4,Y4),(X2, Y2) ]"), dtype = "float32")	# cac diem xep lon xon cung duoc
    paper = four_point_transform(img, pts)
    return paper

def Save_anh(anh,tenf,tmuc):
    import random
    listnn=[1,2,3,4,5,6,7,8,9,0]
    random.shuffle(listnn)
    #tenf = ''
    #for i in listnn[:5]:
    #    tenf = tenf + str(i)
    #tenf = tenf + kitucuoi + '.png'
    cv2.imwrite(tmuc+'/'+tenf, anh)     
    return 

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

def Xem_file_hdan():
    st.subheader(":red[1. Chấm thi trên Phiếu Trắc Nghiệm]")
    chononoff=st.radio("", ('OFF : Xem Hướng Dẫn :', 'ON') ,horizontal=True)
    if chononoff=='ON':
        with open("hdanchamptn.pdf","rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        pass
    st.write("---")
    return


def Lay_dap_an_tu_file_anh(image_file, dic_dap_an): #image_file la object, dic_dap_an ={}
    global SO_DAU_HOI_IN_DA
    pilImg_goc = Image.open(image_file)
    arrImg = np.array(pilImg_goc)
    cv2Img = cv2.cvtColor(arrImg, cv2.COLOR_RGB2BGR)    #mang numpyarray nhung doi sang he mau cua cv2
    cv2Img = cv2.rotate(cv2Img, cv2.ROTATE_90_CLOCKWISE)
    dic_dap_an = Cham_ptn_vanhien_da(cv2Img, dic_dap_an)
    #st.write(dic_dap_an)
    

    #st.write("---")
    return dic_dap_an

def Doi_sang_str(dic_dap_an):
    ch=''
    for i, c in enumerate(dic_dap_an):
        ch = ch + str(i+1) + ':' + dic_dap_an[c] + '---'
    return ch

def Xu_li_dap_an(dic_dap_an):
    global SO_DAU_HOI_IN_DA
    SO_DAU_HOI_IN_DA=0
    ch_da_bl = ''
    ch_dap_an = ''
    for i, c in enumerate(dic_dap_an):
        ch_dap_an = ch_dap_an + str(i+1) + ':' + dic_dap_an[c] + '---'

    for i in range(len(dic_dap_an)):
        if 'K' in dic_dap_an.values():
            SO_DAU_HOI_IN_DA +=1
            if SO_DAU_HOI_IN_DA > 0:
                bluan = 'Đáp án KHÔNG đạt chuẩn!'
                ch_da_bl = ch_dap_an + "\n\n" + bluan    
                return ch_da_bl, SO_DAU_HOI_IN_DA
    if SO_DAU_HOI_IN_DA == 0:        
        bluan = 'Đáp án ĐẠT chuẩn!'
        ch_da_bl = ch_dap_an + "\n\n" + bluan    
        return ch_da_bl, SO_DAU_HOI_IN_DA
    


def Xem_gon_dap_an(dic_dap_an,SO_DAU_HOI_IN_DA,ch_da_bl):
    ch_da_bl = ''
    if dic_dap_an !={}:
        chononoff_da = st.radio("", ('ON : Xem Đáp Án :', 'OFF') ,horizontal=True)
        if chononoff_da == 'OFF':
            return dic_dap_an,SO_DAU_HOI_IN_DA,bluan
        else:
            ch=''
            for i, c in enumerate(dic_dap_an):
                ch = ch + str(i+1) + ':' + dic_dap_an[c] + '---'
            
            
            for i in range(len(dic_dap_an)):
                if 'K' in dic_dap_an.values():
                    SO_DAU_HOI_IN_DA +=1
                    if SO_DAU_HOI_IN_DA > 0:
                        bluan = 'Đáp án KHÔNG đạt chuẩn!'
                        break
            if SO_DAU_HOI_IN_DA == 0:       
                bluan = 'Đáp án ĐẠT chuẩn!'
    
            ch_da_bl= ch+"\n\n" + bluan
            #st.write(dic_dap_an)
    
            return dic_dap_an,SO_DAU_HOI_IN_DA,ch_da_bl


def Cham_ptn_vanhien_hv(image, dic_dap_an):
    print(dic_dap_an)
    cnts_4KV_top, cnts_4KV_bot = Find_4kv_Black_Big_Top_Bot(image)
    paper = Scanhoa_from_4dinh_of4kv_mark(image,cnts_4KV_top, cnts_4KV_bot)
    # Tim cac cnts EXTERNAL tren paper
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    CNT_max=cnts[0] #CNT_max la hcn lon nhat chua 4 khoi bubs
    Xo,Yo,Wo,Ho = cv2.boundingRect(CNT_max)
    #sau nay toado cua cac cnt phai cong them (Xo,Yo)

    #Cat lay anh paper_ptn
    bdaycat=10
    paper_ptn = paper[Yo+bdaycat:Yo+Ho-2*bdaycat, Xo+bdaycat:Xo+Wo-2*bdaycat]   # da xen xq beday 2 de sau do lay cac cnts EXTERNAL vuong nho
    #sau nay toado cua cac cnt phai cong them (Xo+2,Yo+2)

    #brow_img(image_ptn,'image_ptn')
    gray = cv2.cvtColor(paper_ptn, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    #Lay 200 o vuong co arrea lon nhat
    cnts__200bubs = cnts[:200]
    cnts__200bubs = sorted(cnts__200bubs, key=get_x_ver0)

    cnts_200bubs_sxep=[]
    for i in np.arange(0, len(cnts__200bubs), 40):
        cnts_khoi40 = cnts__200bubs[i:i+40]

        cnts_khoi40 = sorted(cnts_khoi40, key=get_y_ver1)
        
        cnts_khoi40_nua=[]
        for j in np.arange(0, len(cnts_khoi40), 4):
            cnts_hang = cnts_khoi40[j:j+4]
            cnts_hang = sorted(cnts_hang, key=get_x_ver0)
            cnts_khoi40_nua = cnts_khoi40_nua + cnts_hang
        
        cnts_200bubs_sxep = cnts_200bubs_sxep + cnts_khoi40_nua

    # Doi toado
    cnts_200bubs_sxep_inpaper=[]
    for cnt in cnts_200bubs_sxep:
        cnts_200bubs_sxep_inpaper.append(cnt + np.array([Xo + bdaycat, Yo + bdaycat]))
    #for cnt in cnts_200bubs_sxep_inpaper:
    #    x,y,w,h = cv2.boundingRect(cnt)
        #cv2.rectangle(paper,(x,y),(x+w,y+h),(0,255,0),2)	
        #cv2.drawContours(image, [cnt], 0, (0,0,255), 1)
    #    brow_img(paper,'Anh goc')
    #print(len(cnts_200bubs_sxep_inpaper))
    #exit()
    # Test
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
    dosang_min = 4000
    dosang_max = 0

    bdaycat=10
    list_dosang=[]
    for i, cnt in enumerate(cnts_200bubs_sxep_inpaper):
        x,y,w,h = cv2.boundingRect(cnt)
        anh = thresh[y+bdaycat:y+h-2*bdaycat, x+bdaycat:x+w-2*bdaycat]
        anh = cv2.resize(anh, (28, 28), cv2.INTER_AREA)
        #anh = anh.reshape((28, 28, 1))
        #print(anh)
        #print(anh.shape)
        #anh = anh//255  # chia lay phan nguyen
        #print(anh)
        total=cv2.countNonZero(anh)
        list_dosang.append(total)
        if total >420:
            clas='3'
        elif total >250:
            clas='2'
        elif total >120:
            clas='1'
        else:
            clas='0'
        if os.path.isdir("Datasets/TAM"):
            tenf=str((i//4)+1)+'_'+str(i%4)+'_'+str(total)+'_'+clas+'.png'
            Save_anh(anh,tenf,tmuc='D:/Deep_learning_030823/Datasets/TAM')
        if total >= dosang_max:
            dosang_max = total
        if total <= dosang_min:
            dosang_min = total
        #brow_img(anh,str(total))
        #cv2.imwrite('dataset/label_1/img_'+str(i)+'.png', anh)
    print(dosang_min,dosang_max)

    caus_voi_dosang_ofABCD = []
    dosang_b=[0,0,0,0]
    for k,c in enumerate(cnts_200bubs_sxep_inpaper):
        idx = k % 4
        #print(idx)
        d_dosang = list_dosang[k]
        #dosang_b[idx] = d_dosang
        #print(str(k+1)+'- do sang : ',d_dosang)
        if (420 < d_dosang):    # [421, +vocuc)
            dosang_b[idx] = 3   
        elif (250 < d_dosang):  # [251,420]
            dosang_b[idx] = 2
        elif (120 < d_dosang):  # [121,250]
            dosang_b[idx] = 1
        else:
            dosang_b[idx] = 0   # [0,120]
        #dosang_b[idx] = d_dosang    
        if idx == 3 and k>=3:
            #print(dosang_b)
            caus_voi_dosang_ofABCD.append(dosang_b)
            #print(dosang_b)
            #brow_img(paper,str(k+1))
            dosang_b=[0,0,0,0]  #khoi tao lai each 4
    #print(len(caus_voi_dosang_ofABCD))
    #print(caus_voi_dosang_ofABCD[-1])
    # Ve KQ vao cnt cho moi cau
    #print('So KV : ',len(all_cnts_kv_bubs_toado_inpaper))
    #print('So Cau hoi : ',len(caus_voi_dosang_ofABCD))
    dic_idx = {'A':0, 'B':1, 'C':2, 'D':3}

    #for cnt in all_cnts_kv_bubs_toado_inpaper:
    #    cv2.drawContours(paper, [cnt], 0, (0,0,255), 2)
    #brow_img(paper,'anhkb')
    #################
    ###############    
    so_cau_dung=0
    for ic,listdosanginABCD in enumerate(caus_voi_dosang_ofABCD):
        #Ve_Kq_Cau(paper, cntve, dapanofcauic, listdosanginABCD)
        #Xet cau ic:
        KITUofDA = dic_dap_an[ic] #ki tu nao (A/B/C/D) cua dap an
        #print(KITUofDA) 
        #listdosanginABCD = listdosanginABCD
        idxofKITUofDA = dic_idx[KITUofDA]
        #print(idxofKITUofDA)
        idxCnt = 4*ic + dic_idx[KITUofDA]   #cau thu may
        #print(idxCnt)
        cnt_de_ve = cnts_200bubs_sxep_inpaper[idxCnt]  # cnt o vi tri dang xet
        #print(cnt_de_ve)
        #print(listdosanginABCD)
        sochidosang = listdosanginABCD[idxofKITUofDA]
        #print(sochidosang)
        xb,yb,wb,hb = cv2.boundingRect(cnt_de_ve)   # hcn bao cnt
        #print(listdosanginABCD.count(1))
        #print(listdosanginABCD.count(3))
        #if (sochidosang == 1 and listdosanginABCD.count(1) == 1 and listdosanginABCD.count(3) == 0) or (sochidosang == 3 and listdosanginABCD.count(3) == 1 and listdosanginABCD.count(1) == 0):
        if (listdosanginABCD.count(1) == 1 and listdosanginABCD.count(3) == 0) or (listdosanginABCD.count(3) == 1 and listdosanginABCD.count(1) == 0):
            cv2.rectangle(paper, (xb,yb),(xb+wb,yb+hb),(0,255,0),3)
            #cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2)-2,(0,255,0),4) # GREEN
            so_cau_dung=so_cau_dung+1
            print(listdosanginABCD)

        elif (listdosanginABCD.count(1) > 1 or listdosanginABCD.count(3) > 1) or (listdosanginABCD.count(1) == 1 and listdosanginABCD.count(3) == 1):
            cv2.rectangle(paper, (xb,yb),(xb+wb,yb+hb),(225,0,225),3)
            #cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2)-2,(255,0,255),4) # PINK
        else:
            cv2.rectangle(paper, (xb,yb),(xb+wb,yb+hb),(0,0,255),3)
            #cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2)-2,(0,0,255),4)  #RED

        #print(ic+1)
        #if ic == 39:
        #    print(listdosanginABCD)
        #    brow_img(paper,'vekq')    
    #brow_img(paper,'vekq')
    if len(caus_voi_dosang_ofABCD) > 0:
        diem = float("{:.2f}".format(10*so_cau_dung/len(caus_voi_dosang_ofABCD)))
        ket_qua_thi = 'Diem : '+str(diem)+' (Ti le cau dung: '+str(so_cau_dung)+'/'+str(len(caus_voi_dosang_ofABCD))+')'
    else:
        print('Sorry! So cau hoi = ',len(caus_voi_dosang_ofABCD))    
        exit()

    # Ghi ket qua vao Phieu
    #paper=cv2.resize(paper,(2026,1325) )
    #paper=cv2.resize(paper,(680,380))	#nua to A4

    so_bao_danh='??????'
    ma_de_thi='???'

    #text1='So bao danh: '+so_bao_danh+' . Ma de thi: '+ma_de_thi
    text1=ket_qua_thi
    #toado1 = (toado_ghidiem[0], toado_ghidiem[1])
    toado1 = (100,70)   #(x,y)
    #toado1 = TOADO_paper_tn_IN_paper
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    color1 = (0, 0,255)
    thickness = 2
    cv2.putText(paper, text1, toado1, font, fontScale, color1, thickness, cv2.LINE_AA)

    #cv2.imwrite("PTN_DA_CHAM/"+so_bao_danh.replace('?','k')+".jpg",paper)

    #print(ket_qua_thi)

    return  paper, ket_qua_thi 

def Cham_ptn_vanhien_da(image, dic_dap_an): # image la cua cv2, dic_dap_an ={}
    dic_dap_an={}
    cnts_4KV_top, cnts_4KV_bot = Find_4kv_Black_Big_Top_Bot(image)
    paper = Scanhoa_from_4dinh_of4kv_mark(image,cnts_4KV_top, cnts_4KV_bot)
    # Tim cac cnts EXTERNAL tren paper
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    CNT_max=cnts[0] #CNT_max la hcn lon nhat chua 4 khoi bubs
    Xo,Yo,Wo,Ho = cv2.boundingRect(CNT_max)
    #sau nay toado cua cac cnt phai cong them (Xo,Yo)

    #Cat lay anh paper_ptn
    bdaycat=10
    paper_ptn = paper[Yo+bdaycat:Yo+Ho-2*bdaycat, Xo+bdaycat:Xo+Wo-2*bdaycat]   # da xen xq beday 2 de sau do lay cac cnts EXTERNAL vuong nho
    #sau nay toado cua cac cnt phai cong them (Xo+2,Yo+2)

    #brow_img(image_ptn,'image_ptn')
    gray = cv2.cvtColor(paper_ptn, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    #Lay 200 o vuong co arrea lon nhat
    cnts__200bubs = cnts[:200]
    cnts__200bubs = sorted(cnts__200bubs, key=get_x_ver0)

    cnts_200bubs_sxep=[]
    for i in np.arange(0, len(cnts__200bubs), 40):
        cnts_khoi40 = cnts__200bubs[i:i+40]

        cnts_khoi40 = sorted(cnts_khoi40, key=get_y_ver1)
        
        cnts_khoi40_nua=[]
        for j in np.arange(0, len(cnts_khoi40), 4):
            cnts_hang = cnts_khoi40[j:j+4]
            cnts_hang = sorted(cnts_hang, key=get_x_ver0)
            cnts_khoi40_nua = cnts_khoi40_nua + cnts_hang
        
        cnts_200bubs_sxep = cnts_200bubs_sxep + cnts_khoi40_nua

    # Doi toado
    cnts_200bubs_sxep_inpaper=[]
    for cnt in cnts_200bubs_sxep:
        cnts_200bubs_sxep_inpaper.append(cnt + np.array([Xo + bdaycat, Yo + bdaycat]))
    #for cnt in cnts_200bubs_sxep_inpaper:
    #    x,y,w,h = cv2.boundingRect(cnt)
        #cv2.rectangle(paper,(x,y),(x+w,y+h),(0,255,0),2)	
        #cv2.drawContours(image, [cnt], 0, (0,0,255), 1)
    #    brow_img(paper,'Anh goc')
    #print(len(cnts_200bubs_sxep_inpaper))
    #exit()
    # Test
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
    dosang_min = 4000
    dosang_max = 0

    bdaycat=10
    list_dosang=[]
    for i, cnt in enumerate(cnts_200bubs_sxep_inpaper):
        x,y,w,h = cv2.boundingRect(cnt)
        anh = thresh[y+bdaycat:y+h-2*bdaycat, x+bdaycat:x+w-2*bdaycat]
        anh = cv2.resize(anh, (28, 28), cv2.INTER_AREA)
        #anh = anh.reshape((28, 28, 1))
        #print(anh)
        #print(anh.shape)
        #anh = anh//255  # chia lay phan nguyen
        #print(anh)
        total=cv2.countNonZero(anh)
        list_dosang.append(total)
        if total >420:
            clas='3'
        elif total >250:
            clas='2'
        elif total >120:
            clas='1'
        else:
            clas='0'
        if os.path.isdir("Datasets/TAM"):
            tenf=str((i//4)+1)+'_'+str(i%4)+'_'+str(total)+'_'+clas+'.png'
            #Save_anh(anh,tenf,tmuc='D:/Deep_learning_030823/Datasets/TAM')
        if total >= dosang_max:
            dosang_max = total
        if total <= dosang_min:
            dosang_min = total
        #brow_img(anh,str(total))
        #cv2.imwrite('dataset/label_1/img_'+str(i)+'.png', anh)
    #print(dosang_min,dosang_max)

    caus_voi_dosang_ofABCD = []
    dosang_b=[0,0,0,0]
    for k,c in enumerate(cnts_200bubs_sxep_inpaper):
        idx = k % 4
        #print(idx)
        d_dosang = list_dosang[k]
        #dosang_b[idx] = d_dosang
        #print(str(k+1)+'- do sang : ',d_dosang)
        if (420 < d_dosang):    # [421, +vocuc)
            dosang_b[idx] = 3   
        elif (250 < d_dosang):  # [251,420]
            dosang_b[idx] = 2
        elif (120 < d_dosang):  # [121,250]
            dosang_b[idx] = 1
        else:
            dosang_b[idx] = 0   # [0,120]
        #dosang_b[idx] = d_dosang    
        if idx == 3 and k>=3:
            #print(dosang_b)
            caus_voi_dosang_ofABCD.append(dosang_b)
            #print(dosang_b)
            #brow_img(paper,str(k+1))
            dosang_b=[0,0,0,0]  #khoi tao lai each 4
    #print(len(caus_voi_dosang_ofABCD))
    #print(caus_voi_dosang_ofABCD[-1])
    # Ve KQ vao cnt cho moi cau
    #print('So KV : ',len(all_cnts_kv_bubs_toado_inpaper))
    #print('So Cau hoi : ',len(caus_voi_dosang_ofABCD))
    #for cnt in all_cnts_kv_bubs_toado_inpaper:
    #    cv2.drawContours(paper, [cnt], 0, (0,0,255), 2)
    #brow_img(paper,'anhkb')
    #################
    #so_cau_dung=0
    for ic,listdosanginABCD in enumerate(caus_voi_dosang_ofABCD):
        if (listdosanginABCD[0]==1 or listdosanginABCD[0]==3) and ((listdosanginABCD.count(1) == 1 and listdosanginABCD.count(3) == 0) or (listdosanginABCD.count(3) == 1 and listdosanginABCD.count(1) == 0)):
            kt_dap='A'
        elif (listdosanginABCD[1]==1 or listdosanginABCD[1]==3) and ((listdosanginABCD.count(1) == 1 and listdosanginABCD.count(3) == 0) or (listdosanginABCD.count(3) == 1 and listdosanginABCD.count(1) == 0)):
            kt_dap='B'
        elif (listdosanginABCD[2]==1 or listdosanginABCD[2]==3) and ((listdosanginABCD.count(1) == 1 and listdosanginABCD.count(3) == 0) or (listdosanginABCD.count(3) == 1 and listdosanginABCD.count(1) == 0)):
            kt_dap='C'
        elif (listdosanginABCD[3]==1 or listdosanginABCD[3]==3) and ((listdosanginABCD.count(1) == 1 and listdosanginABCD.count(3) == 0) or (listdosanginABCD.count(3) == 1 and listdosanginABCD.count(1) == 0)):
            kt_dap='D'
        else:
            kt_dap='K'
        dic_dap_an[ic] = kt_dap
    return dic_dap_an

def Chuyen_anh_cv2pil(paper, ket_qua_thi):

    st.write("TTTTTTTTTTTTTTTTTTT")
    return
    '''
    img = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    # For reversing the operation:
    pil_im_pil = np.asarray(im_pil)

    #dung ham exif_transpose(imggocinPil) cua modul ImageOps in PIL de xoay lai anh trong st.image
    #pil_im_pil = ImageOps.exif_transpose(im_pil)
    st.image(pil_im_pil, caption='Phiếu trắc nghiệm đã được chấm!')

    st.write(":red["+ket_qua_thi+"]")
    return
    '''
