# reboot app : https://share.streamlit.io/
#lenh chay
#streamlit run streamlit_app.py
#pip install streamlit-camera-input-live
import cv2
import streamlit as st
import numpy as np
import os
import pickle
#from streamlit_option_menu import option_menu
from PIL import Image
from PIL import ImageOps
#from PIL import ImageGrab  
import imutils
#from imutils.perspective import four_point_transform
global dic_dap_an,mau_phieu_chon
dic_dap_an={}
mau_phieu_chon=''
##########################
def brow_img(image,namewin):
    cv2.namedWindow(namewin, cv2.WINDOW_NORMAL) 
    cv2.imshow(namewin, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def get_x_ver0(s):
   s = cv2.boundingRect(s)
   return s[0]

def get_y_ver1(s):
   s = cv2.boundingRect(s)
   return s[1]

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
        if 'dap_an_' in tep and '.pkl' in tep:
            ltep.append(tep[:-4])
    ltep.sort(reverse=True)
    option = st.selectbox(
        ':blue[Chọn một đáp án đã có trước đây?]',
        (ltep[i] for i in range(len(ltep))))

    tepluuda = './dap_an/' + option + '.pkl'
    #st.write(tenfda_dang_dung)

    with open(tepluuda, 'rb') as fwb:
        ldata = pickle.load(fwb)
        dic_dap_an = ldata[0]
        ch_da = ldata[1]
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

            dic_dap_an, ch_da = Cham_ptn_vanhien_da(cv2Img)

            if dic_dap_an != {}:
                ch_da = ''
                for keyd in dic_dap_an:
                    if  keyd < len(dic_dap_an) - 1 :  
                        ch_da = ch_da + str(keyd+1) + dic_dap_an[keyd] + ', '
                    else:    
                        ch_da = ch_da + str(keyd+1) + dic_dap_an[keyd]
                #st.write(ch_dap_an)
                tepluub = "dap_an/" + get_ten_file()
                #tepluub = tepluub.replace('.jpg','dap_an_'+made+'.pkl')
                with open(tepluub, 'wb') as fwb:
                    ldap_an=[dic_dap_an,ch_da]
                    pickle.dump(ldap_an, fwb)
                    st.write('Đã lưu đáp án!')    
                    #return ch_da
            else:
                st.write('Cung cấp đáp án mới chưa thành công!')
    return dic_dap_an,ch_da
    
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

def Cham_ptn_vanhien_da(image): # image la cua cv2, dic_dap_an ={}
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
        #if os.path.isdir("Datasets/TAM"):
        #    tenf=str((i//4)+1)+'_'+str(i%4)+'_'+str(total)+'_'+clas+'.png'
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
            exit('PTN DAP AN CO LOI!')
        dic_dap_an[ic] = kt_dap
    ch_da = ''
    for keyd in dic_dap_an:
        if  keyd < len(dic_dap_an) - 1 :  
            ch_da = ch_da + str(keyd+1) + dic_dap_an[keyd] + ', '
        else:    
            ch_da = ch_da + str(keyd+1) + dic_dap_an[keyd]
    st.write(ch_da)
    tepluub = get_ten_file()
    tepluub = tepluub.replace('.jpg','dap_an_.pkl')
    with open(tepluub, 'wb') as fwb:
        ldap_an=[dic_dap_an,ch_da]
        pickle.dump(ldap_an, fwb)
        st.write('Đã lưu đáp án!')    
    return dic_dap_an, ch_da

def get_ten_file():
    masoda_da_nhap = st.text_input('Nhập mã số của đáp án với 3 kí tự số : ', '   ', max_chars=3)
    st.write('Mã số của đáp án đã nhập là : ', masoda_da_nhap)
    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    ch=dt_string.replace('/','_')
    ch=ch.replace(' ','_')
    ch=ch.replace(':','')
    return ch + 'dap_an_'+masoda_da_nhap+'.pkl'

def check_4kvd_in_img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
    cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
    if len(cnts)<5:
        return "PTN khong chuan"
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
        return "PTN khong chuan"        
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
        return "PTN khong chuan"       
    cnts_lay = sorted(cnts_lay,key=get_y_ver1)
    cnt_4c_TOP = cnts_lay[:2]
    cnt_4c_BOT = cnts_lay[-2:] 
    cnt_4c_TOP = sorted(cnt_4c_TOP, key= get_x_ver0)
    cnt_4c_BOT = sorted(cnt_4c_BOT, key= get_x_ver0)
    #paper = Scanhoa_from_4dinh_of4kv_mark(image,cnt_4c_TOP, cnt_4c_BOT)
    st.write("PTN dat chuan")
    return cnt_4c_TOP, cnt_4c_BOT

def Cham_ptn_qua_camera(dic_dap_an):
    if dic_dap_an == {}:
        return st.write('Không thể làm việc vì chưa cung cấp đáp án!')    

    img_file_buffer = st.camera_input("Chụp Phiếu Trắc Nghiệm (nhấp vào Take Photo) sao cho có 4 khối đen trong PTN ở 4 góc ảnh",key='CPTNQC')

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        paper, ket_qua_thi = Cham_ptn_vanhien_hv(cv2_img, dic_dap_an)
        st.image(paper)
        st.write(ket_qua_thi)
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

def Cham_ptn_vanhien_hv(image, dic_dap_an):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
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
        #if os.path.isdir("Datasets/TAM"):
        #    tenf=str((i//4)+1)+'_'+str(i%4)+'_'+str(total)+'_'+clas+'.png'
        #    Save_anh(anh,tenf,tmuc='D:/Deep_learning_030823/Datasets/TAM')
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
            #print(listdosanginABCD)

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

def Upload_fileimg_ptn():
    if dic_dap_an == {}:
        return st.write('Không thể làm việc vì chưa cung cấp đáp án!')    
    
    image_file = st.file_uploader(":green[Chọn 1 file ảnh Phiếu Trăc Nghiệm để tải lên.]",type=("png", "jpg"), key=4)

    if image_file is not None:
        #st.write(image_file.name + ' đã được tải lên')

        pilImg_goc = Image.open(image_file)
        arrImg = np.array(pilImg_goc)
        cv2Img = cv2.cvtColor(arrImg, cv2.COLOR_RGB2BGR)    #mang numpyarray nhung doi sang he mau cua cv2
        cv2Img = cv2.rotate(cv2Img, cv2.ROTATE_90_CLOCKWISE)
        paper, ket_qua_thi = Cham_ptn_vanhien_hv(cv2Img, dic_dap_an)
        
        st.write("Chấm xong. Dưới đây là ảnh PTN đã chấm:")

        img = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        # For reversing the operation:
        pil_im_pil = np.asarray(im_pil)

        #dung ham exif_transpose(imggocinPil) cua modul ImageOps in PIL de xoay lai anh trong st.image
        pil_im_pil = ImageOps.exif_transpose(im_pil)

        st.image(pil_im_pil, caption='Phiếu trắc nghiệm đã được chấm!')

        st.write(":red["+ket_qua_thi+"]")

        from datetime import datetime
        now = datetime.now()
        dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
        ch=dt_string.replace('/','_')
        ch=ch.replace(' ','_')
        ch=ch.replace(':','')
        str_tep = ch +'.jpg'

        from io import BytesIO
        buf = BytesIO()
        pil_im_pil.save(buf, format="JPEG") # pil_im_pil la np.aray trong PIL cua anh tren
        byte_im = buf.getvalue()
        #use the st.download_button
        btn = st.download_button(
            label = ":blue[Download Phiếu trắc nghiệm với kết quả vừa được chấm!]",
            data = byte_im,
            file_name = str_tep,
            mime="image/jpeg",
            )


################################################################################################
# main()


st.title("Chấm Phiếu Trắc Nghiệm auto online")
if st.checkbox('**:red[Bước 1 : Chọn mẫu phiếu]**'):
    mau_phieu_chon = Chon_mau_phieu()
    st.write('Mẫu phiếu đã chọn là : '+mau_phieu_chon)

if st.checkbox('**:red[Bước 2 : Cung cấp đáp án]**'):
    if mau_phieu_chon !='':
        dic_dap_an, ch_da = Cung_cap_da()
    else:    
        st.write('Chưa chọn mẫu phiếu TN nên không thể làm việc!')

if st.checkbox('**:red[Bước 3 : Upload file image PTN trong máy lên, sau đó auto chấm rồi trả về kết quả.]**'):
    Upload_fileimg_ptn()

if st.checkbox('**:red[Phụ lục : Chụp PTN bằng camera online và xử lí auto]**'):
    Cham_ptn_qua_camera(dic_dap_an)
