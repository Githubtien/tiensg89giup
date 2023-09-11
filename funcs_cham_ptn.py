#Modul nay chua cac ham ho tro cham_ptn
import cv2
import numpy as np
import os
import pickle
from PIL import Image
from PIL import ImageOps
import imutils
import random

#1-Tao_dicdapan_random(socau)
#2-brow_img(image,namewin)
#3-order_points(pts)
#4-four_point_transform(image, pts)
#5-get_x_ver0(s)
#6-get_y_ver1(s)
#7-Xen_xquanh_anh(image,beday)
#8-Xen_trai_anh(paper_tn,beday)
#9-Scanhoa_from_4dinh_of4kv_mark(img,cnts_4KV_top, cnts_4KV_bot)
#10-Xu_li_bub_tinh_diem_thi(All_cnts_bub_in_paper, paper,dic_dap_an)
#11-get_ten_file_time()
#12-save_dap_an_pkl(ma_de, dic_dap_an)
#13-Find_4kv_va_scanhoa(image)
#14-Scan_hoa_theo_hcn_baocnt(image,cnt_x)
#15-Sap_xeptt_allbubs(cnts_bubs,so_khoi,so_bub_moi_khoi,so_cau_moi_khoi,so_bub_moi_cau)
#16-Find_cnts_voi_kieuN(image,kieuN):

##########################
# dung de xem anh cho tien
def Tao_dicdapan_random(socau):
    dic_dap_an={}
    lnnlamda=['A','B','C','D']
    for i in range(socau):
        random.shuffle(lnnlamda)
        dic_dap_an[i]=lnnlamda[i%4]
    return dic_dap_an   

def brow_img(image,namewin):
    cv2.namedWindow(namewin, cv2.WINDOW_NORMAL) 
    cv2.imshow(namewin, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return

# sap xep lai thu tu 4 dinh cua hcn de scan hoa
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

# scan hoa theo 4 dinh hcn
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

# sort cnts theo truc x
def get_x_ver0(s):
   s = cv2.boundingRect(s)
   return s[0]
# sort cnts theo truc y
def get_y_ver1(s):
   s = cv2.boundingRect(s)
   return s[1]
# Xen anh xq
def Xen_xquanh_anh(image,beday):
	anh_sau_xen=image[beday:image.shape[0]-2*beday,beday:image.shape[1]-2*beday]
	return anh_sau_xen
# xen ben trai anh
def Xen_trai_anh(paper_tn,beday):
    paper_tn = paper_tn[:paper_tn.shape[0],beday:paper_tn.shape[1]]
    return paper_tn


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

def Xu_li_bub_tinh_diem_thi(All_cnts_bub_in_paper, paper,dic_dap_an):
    answer_choices = ['A', 'B', 'C', 'D', '?'] 
    test_sensitivity_epsilon=10
    questions_answer=[]

    warped = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    means=[]
    cau=0
    so_cau_dung=0
    for i,c in enumerate(All_cnts_bub_in_paper):
        (x, y, w, h) = cv2.boundingRect(c)
        anh = warped[y:y+h, x:x+w]
        #print(np.mean(anh))
        means.append(int(np.mean(anh)))
        idx_answ = i % 4
        cau=int(i/4)
        #find image means of the answer bubbles
        if len(means)==4 :
            #print(str(cau)+':',means)
            #brow_img(warped,'warped')
            #sort by minimum mean; sort by the darkest bubble
            min_arg = np.argmin(means)
            min_val = means[min_arg]
            #find the second smallest mean
            means[min_arg] = 255
            min_val2 = means[np.argmin(means)]
            #check if the two smallest values are close in value
            if min_val2 - min_val < test_sensitivity_epsilon:
                #if so, then the question has been double bubbled and is invalid
                min_arg = 4
            #append the answers to the array
            questions_answer.append(answer_choices[min_arg])
            # To mau cnt
            # neu answer_choices[min_arg] = A thi ung voi cnt: cau*4 + i % 4 
            if answer_choices[min_arg] == dic_dap_an[cau]:
                xb,yb,wb,hb = cv2.boundingRect(All_cnts_bub_in_paper[cau*4 + min_arg])
                cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2),(0,255,0),4) # GREEN #bk cong them 2 cho ro
                so_cau_dung = so_cau_dung+1
                #brow_img(paper,'paper')
            else:
                if answer_choices[min_arg]=='?':
                    # Lay idx cua dap an cau nay
                    kitu = dic_dap_an[cau]	# A hoac B hoac C hoac D
                    idx = answer_choices.index(kitu)
                    xb,yb,wb,hb = cv2.boundingRect(All_cnts_bub_in_paper[cau*4 + idx])
                    cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2),(225,0,225),4) # PRINK
                #	brow_img(paper,'paper'))
                else:
                    kitu = dic_dap_an[cau]	# A hoac B hoac C hoac D
                    idx = answer_choices.index(kitu)
                    xb,yb,wb,hb = cv2.boundingRect(All_cnts_bub_in_paper[cau*4 + idx])
                    cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2),(0,0,255),4) # RED
                #	brow_img(paper,'paper')
            means = []
    #ket_qua_thi='pppppp'
    if len(questions_answer)>0:
        diem = float("{:.2f}".format(10*so_cau_dung/len(questions_answer)))
        #print(str(diem))
        #ket_qua_thi = 'Diem : '+str(10*round(so_cau_dung/len(questions_answer),3))+' (Ti le cau dung: '+str(so_cau_dung)+'/'+str(len(questions_answer))+')'
        #ket_qua_thi = 'Diem : '+str(diem)+' (Ti le cau dung: '+str(so_cau_dung)+'/'+str(len(questions_answer))+')'
        lkqthi=[diem,so_cau_dung,len(questions_answer)]
    else:
        lkqthi=[0,0,0]    
    #brow_img(paper,'paper')
    return paper,lkqthi

# xu li khoi bao danh

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

def save_dap_an_pkl(ma_de, dic_dap_an):
    tepluub = 'dap_an/'+get_ten_file_time() + '_da_' + ma_de + '_' + socau + '.pkl'
    with open(tepluub, 'wb') as fwb:
        pickle.dump(dic_dap_an, fwb)
    print('Da luu dic dap an vao file:', tepluub)
    return tepluub

def Find_4kv_va_scanhoa(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)
    cnts_lay=[]
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)    
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 :
            cnts_lay.append(cnt)
    if len(cnts_lay)<4:
        exit('Khong co 4 KV')
    cnts_cac_kv = sorted(cnts_lay, key= get_y_ver1)
    cnt_4c_TOP = cnts_cac_kv[:2]   # lay 2 cnt dau tien hy vng la 2 kv
    cnt_4c_BOT = cnts_cac_kv[-2:] # lay 2 cnt cuoi cung  hy vng la 2 kv
    cnt_4c_TOP = sorted(cnt_4c_TOP, key= get_x_ver0)
    cnt_4c_BOT = sorted(cnt_4c_BOT, key= get_x_ver0)
    paper = Scanhoa_from_4dinh_of4kv_mark(image,cnt_4c_TOP, cnt_4c_BOT)
    return paper

def Find_cnts_voi_kieuN(image,kieuN):
    # kieuN = 0,1,2,3,4 ung voi cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP, cv2.ETR_TREE, cv2.RETR_FLOODFILL 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, kieuN, cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)
    return cnts

def Scan_hoa_theo_hcn_baocnt(image,cnt_x):
    xM,yM,wM,hM =cv2.boundingRect(cnt_x)
    X1,Y1 = xM, yM
    X2,Y2 = xM+wM, yM
    X3,Y3 = xM,yM+hM
    X4,Y4 = xM+wM,yM+hM
    pts = np.array(eval("[(X1, Y1), (X3,Y3),(X4,Y4),(X2, Y2) ]"), dtype = "float32")	# cac diem xep lon xon cung duoc
    paper = four_point_transform(image, pts)
    return paper

def Sap_xeptt_allbubs(cnts_bubs,so_khoi,so_bub_moi_khoi,so_cau_moi_khoi,so_bub_moi_cau):
    cnts_all_bubs_sx=[]
    for iK in range(so_khoi):
        khoi_120bubs = cnts_bubs[(iK%so_khoi)*so_bub_moi_khoi :(iK%so_khoi)*so_bub_moi_khoi+so_bub_moi_khoi]
        khoi_120bubs = sorted(khoi_120bubs,key=get_y_ver1)
        for jC in range(so_cau_moi_khoi):
            dong_4bubs = khoi_120bubs[(jC%so_cau_moi_khoi)*so_bub_moi_cau :(jC%so_cau_moi_khoi)*so_bub_moi_cau + so_bub_moi_cau]
            dong_4bubs = sorted(dong_4bubs, key=get_x_ver0)
            #dong_4bubs_toadocu=[]
            #for cc in dong_4bubs:
            #    dong_4bubs_toadocu.append(cc+np.array([xM+4,yM+4])) 
            #cnts_all_bubs_sx = cnts_all_bubs_sx + dong_4bubs_toadocu
            cnts_all_bubs_sx = cnts_all_bubs_sx + dong_4bubs
    return cnts_all_bubs_sx

######################################################################d##########################
