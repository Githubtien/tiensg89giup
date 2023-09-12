import numpy as np
import cv2
import imutils
import os.path
#from funcs_cham_ptn import Lay_so_bao_danh
from funcs_cham_ptn import Xen_xquanh_anh
from funcs_cham_ptn import get_x_ver0
from funcs_cham_ptn import get_y_ver1
#from funcs_cham_ptn import Xu_li_bub_tinh_diem_thi
from funcs_cham_ptn import Tao_dicdapan_random
from funcs_cham_ptn import brow_img
from funcs_cham_ptn import Find_cnts_voi_kieuN
from funcs_cham_ptn import four_point_transform

def Lay_so_bao_danh(paper):
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # 1-Lay SO BAO DANH 
    cnts_khoi_bd = cnts[2]  #hcn idx 2 la bd
    x,y,w,h = cv2.boundingRect(cnts_khoi_bd)
    anh_khoi_bd=paper[y:y+h,x:x+w]
    anh_khoi_bd = Xen_xquanh_anh(anh_khoi_bd,beday=10)
    
    gray = cv2.cvtColor(anh_khoi_bd, cv2.COLOR_BGR2GRAY) 
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts_kvlay=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(approx) != 4 : 
            cnts_kvlay.append(c)
            #cv2.rectangle(anh_khoi_bd,(x,y),(x+w,y+h),(0,255,0),2)	
    #brow_img(anh_khoi_bd,str(len(cnts_kvlay)))
    #cv2.drawContours(anh_khoi_bd, cnts_kvlay[:60], -1, (0,0,255), 4)
    if len(cnts_kvlay) < 60:
        exit('Khong du 60 bubs!')
    cnts_bulay=cnts_kvlay[:60]
    cnts_bulay = sorted(cnts_bulay, key=get_x_ver0)
    cnts_bulay_sx=[]
    for j in range(6):
        cnts_moicot=cnts_bulay[(j%6)*10 :(j%6)*10+10]
        cnts_moicot=sorted(cnts_moicot,key=get_y_ver1)
        cnts_bulay_sx=cnts_bulay_sx+cnts_moicot

    str_sobd=''
    means=[]
    for c in cnts_bulay_sx: #60 pt
        x,y,w,h = cv2.boundingRect(c)
        anh_hcnbao_cnt=gray[y:y+h,x:x+w]
        anh_hcnbao_cnt=Xen_xquanh_anh(anh_hcnbao_cnt,beday=2)
        #brow_img(anh_hcnbao_cnt,'XXX')
        means.append(int(np.mean(anh_hcnbao_cnt)))
        if len(means)==10:
            min_arg=np.argmin(means)
            min_val=means[min_arg]
            means[min_arg]=255
            min_val2=means[np.argmin(means)]
            #print(min_val2-min_val)
            if min_val2 - min_val < 10: 
                kitulay='?'
            else:
                kitulay=str(min_arg)
            str_sobd=str_sobd+kitulay
            means=[]
    #print(str_sobd)
    return str_sobd            

def Lay_so_ma_de(paper):
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #cv2.drawContours(paper, cnts, -1, (0,0,255), 4)
    #brow_img(paper, 'XXXXXX')
    # 2-Lay SO MA DE 
    cnts_khoi_md = cnts[4]  #hcn idx 4 la md
        
    x,y,w,h = cv2.boundingRect(cnts_khoi_md)
    anh_khoi_md=paper[y:y+h,x:x+w]
    anh_khoi_md = Xen_xquanh_anh(anh_khoi_md,beday=10)
    
    gray = cv2.cvtColor(anh_khoi_md, cv2.COLOR_BGR2GRAY) 
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts_kvlay=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(approx) != 4 : 
            cnts_kvlay.append(c)
            #cv2.rectangle(anh_khoi_bd,(x,y),(x+w,y+h),(0,255,0),2)	
    #brow_img(anh_khoi_bd,str(len(cnts_kvlay)))
    #cv2.drawContours(anh_khoi_bd, cnts_kvlay[:60], -1, (0,0,255), 4)
    if len(cnts_kvlay) < 30:
        exit('Khong du 30 bubs!')
    cnts_bulay=cnts_kvlay[:30]
    cnts_bulay = sorted(cnts_bulay, key=get_x_ver0)
    cnts_bulay_sx=[]
    for j in range(3):
        cnts_moicot=cnts_bulay[(j%3)*10 :(j%3)*10+10]
        cnts_moicot=sorted(cnts_moicot,key=get_y_ver1)
        cnts_bulay_sx=cnts_bulay_sx+cnts_moicot

    str_somd=''
    means=[]
    for c in cnts_bulay_sx: #30 pt
        x,y,w,h = cv2.boundingRect(c)
        anh_hcnbao_cnt=gray[y:y+h,x:x+w]
        anh_hcnbao_cnt=Xen_xquanh_anh(anh_hcnbao_cnt,beday=2)
        #brow_img(anh_hcnbao_cnt,'XXX')
        #print(int(np.mean(anh_hcnbao_cnt)))
        means.append(int(np.mean(anh_hcnbao_cnt)))
        if len(means)==10:
            min_arg=np.argmin(means)
            min_val=means[min_arg]
            means[min_arg]=255
            min_val2=means[np.argmin(means)]
            if min_val2 - min_val < 10: 
                kitulay='?'
            else:
                kitulay=str(min_arg)
            str_somd=str_somd+kitulay
            means=[]
    #print(str_somd)
    return str_somd            

def Sap_xeptt_allbubs(cnts_bubs,so_khoi,so_bub_moi_khoi,so_cau_moi_khoi,so_bub_moi_cau):
    cnts_all_bubs_sx=[]
    for iK in range(so_khoi):
        khoi_120bubs = cnts_bubs[(iK%so_khoi)*so_bub_moi_khoi :(iK%so_khoi)*so_bub_moi_khoi+so_bub_moi_khoi]
        khoi_120bubs = sorted(khoi_120bubs,key=get_y_ver1)
        for jC in range(so_cau_moi_khoi):
            dong_4bubs = khoi_120bubs[(jC%so_cau_moi_khoi)*so_bub_moi_cau :(jC%so_cau_moi_khoi)*so_bub_moi_cau + so_bub_moi_cau]
            dong_4bubs = sorted(dong_4bubs, key=get_x_ver0)
            cnts_all_bubs_sx = cnts_all_bubs_sx + dong_4bubs
    return cnts_all_bubs_sx

def Scan_hoa_theo_hcn_baocnt(image,cnt_x):
    xM,yM,wM,hM =cv2.boundingRect(cnt_x)
    X1,Y1 = xM, yM
    X2,Y2 = xM+wM, yM
    X3,Y3 = xM,yM+hM
    X4,Y4 = xM+wM,yM+hM
    pts = np.array(eval("[(X1, Y1), (X3,Y3),(X4,Y4),(X2, Y2) ]"), dtype = "float32")	# cac diem xep lon xon cung duoc
    paper = four_point_transform(image, pts)
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

def cham_ptn_999_120(image,dic_dap_an):
    # Bo ria lay hcn lon nhat chua noi dung
    cnts = Find_cnts_voi_kieuN(image,kieuN=0)
    cnts=sorted(cnts,key=cv2.contourArea,reverse=True)

    #cv2.drawContours(image, [cnts[0]], 0, (0,0,255), 4)
    #brow_img(image,'xxxx')
    #cv2.drawContours(anh_khoi_bd, cnts_kvlay[:60], -1, (0,0,255), 4)

    cnt_x = cnts[0] #cnt_x la cnt hcn bao boc noi dung ,bo ria

    paper = Scan_hoa_theo_hcn_baocnt(image,cnt_x)
    paper = Xen_xquanh_anh(paper,beday=40)
    #brow_img(paper,'xxxx')

    # Lay so bao danh
    str_sobd = Lay_so_bao_danh(paper)
    #print(str_sobd)

    # Lay ma de
    str_somd = Lay_so_ma_de(paper)
    #print(str_somd)

    # Cat ra anh phan trac nghiem
    cnts = Find_cnts_voi_kieuN(paper,kieuN=0)
    cnts = sorted(cnts,key=cv2.contourArea, reverse=True)
    xM,yM,wM,hM =cv2.boundingRect(cnts[0])  # cnt co dt max la phan trac nghiem
    
    anh_phan_tn = paper[yM:yM+hM,xM:xM+wM] 
    anh_phan_tn = Xen_xquanh_anh(anh_phan_tn,beday=20)

    # Lay cac cnt bubs 
    cnts = Find_cnts_voi_kieuN(anh_phan_tn,kieuN=0)
    cnts = sorted(cnts,key=cv2.contourArea, reverse=True)
    cnt_bubs=[]
    for cnt in cnts:
        x1,y1,w1,h1 = cv2.boundingRect(cnt)
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        if (0.8 <= float(w1/h1) <= 1.2) and len(approx) > 4 and w1 >= 50: #and (30 <= cv2.contourArea(cnt)):
            cnt_bubs.append(cnt)
            #print(w1)
            #cv2.drawContours(anh_phan_tn, [cnt], 0, (0,0,255), 4)
    #brow_img(anh_phan_tn,'xxxx')

    #print(len(cnt_bubs))
    if len(cnt_bubs) != 480:
        exit('PTN khong dat chuan!')

    cnts_bubs = sorted(cnt_bubs, key=get_x_ver0)
    cnts_all_bubs_sx = Sap_xeptt_allbubs(cnts_bubs,so_khoi=4,so_bub_moi_khoi=120,so_cau_moi_khoi=30,so_bub_moi_cau=4)

    # dem ve toadocu
    cnts_all_bubs_sx_inpaper=[]
    for cc in cnts_all_bubs_sx:
        cnts_all_bubs_sx_inpaper.append(cc+np.array([xM+20,yM+20])) 

    # Xu li bubs tinh diem thi
    #dic_dap_an = Tao_dicdapan_random(120)
    ########################################
    All_cnts_bub_in_paper = cnts_all_bubs_sx_inpaper[:4*len(dic_dap_an)]
    paper, lkqthi = Xu_li_bub_tinh_diem_thi(All_cnts_bub_in_paper, paper,dic_dap_an)
    # 2- Ghi ket qua vao PTN va Save anh PTN
    #text1=ket_qua_thi
    #toado1 = (25,18)
    toado1 = (52,30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.6
    color1 = (255,0,0)
    color2 = (0, 0,255)
    thickness = 3
    cv2.putText(paper, 'SBD:',   (50,300), 0, fontScale, color1, thickness, cv2.LINE_AA)
    cv2.putText(paper, str_sobd, (50,360), 0, fontScale, color1, thickness, cv2.LINE_AA)

    cv2.putText(paper, 'MD:',   (50,500), 0, fontScale, color1, thickness, cv2.LINE_AA)
    cv2.putText(paper, str_somd, (50,560), 0, fontScale, color1, thickness, cv2.LINE_AA)

    cv2.putText(paper, 'TLD:',   (50,700), 0, fontScale, color2, thickness, cv2.LINE_AA)
    cv2.putText(paper, str(lkqthi[1])+'/'+str(lkqthi[2]), (50,760), 0, fontScale, color2, thickness, cv2.LINE_AA)

    cv2.putText(paper, 'DIEM:',   (50,900), 0, fontScale, color2, thickness, cv2.LINE_AA)
    cv2.putText(paper, str(lkqthi[0]), (50,960), 0, fontScale, color2, thickness, cv2.LINE_AA)

    cv2.imwrite(str_sobd.replace('?','X')+'_'+str_somd.replace('?','0')+".jpg",paper)
    #print(lkqthi)
    return paper

#############################
#tep="PTN_HS/ptn-hs-002.JPG"
#if not os.path.exists(tep):
#    exit('Khong co tep : '+tep)

#image=cv2.imread(tep)

#dic_dap_an = Tao_dicdapan_random(socau=120)
#paper = cham_ptn_999_120(image,dic_dap_an)
#brow_img(image, 'XXXXXX')
#paper = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#paper = cv2.rotate(paper, cv2.ROTATE_90_CLOCKWISE)
#brow_img(paper, 'XXXXXX')
#cv2.imwrite("ptn-moi.jpg",paper)

