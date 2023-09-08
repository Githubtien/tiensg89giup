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
#16-Find_cnts_voi_kieuN(image,kieuN)

import numpy as np
import cv2
import imutils
#from funcs_cham_ptn import Lay_so_bao_danh
#from funcs_cham_ptn import Lay_so_ma_de
from funcs_cham_ptn import Xen_xquanh_anh
#from funcs_cham_ptn import Xen_trai_anh
from funcs_cham_ptn import get_x_ver0
from funcs_cham_ptn import get_y_ver1
#sfrom funcs_cham_ptn import Xu_li_bub_tinh_diem_thi
from funcs_cham_ptn import Tao_dicdapan_random
from funcs_cham_ptn import brow_img
#from funcs_cham_ptn import Lay_4kvtheoy_scanhoa 
from funcs_cham_ptn import Find_cnts_voi_kieuN
from funcs_cham_ptn import Find_4kv_va_scanhoa


def Xu_li_bub_tinh_diem_thiT(All_cnts_bub_in_paper, paper,dic_dap_an):
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
                cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2),(0,255,0),2) # GREEN #bk cong them 2 cho ro
                so_cau_dung = so_cau_dung+1
                #brow_img(paper,'paper')
            else:
                if answer_choices[min_arg]=='?':
                    # Lay idx cua dap an cau nay
                    kitu = dic_dap_an[cau]	# A hoac B hoac C hoac D
                    idx = answer_choices.index(kitu)
                    xb,yb,wb,hb = cv2.boundingRect(All_cnts_bub_in_paper[cau*4 + idx])
                    cv2.rectangle(paper,(xb,yb),(xb+wb,yb+hb),(225,0,225),4)	
                    #cv2.circle(paper,(xb+int(wb/2),yb+int(hb/2)),int(hb/2),(225,0,225),2) # PRINK

                    #cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(hb/2),(225,0,225),10) # PRINK
                #	brow_img(paper,'paper'))
                else:
                    kitu = dic_dap_an[cau]	# A hoac B hoac C hoac D
                    idx = answer_choices.index(kitu)
                    xb,yb,wb,hb = cv2.boundingRect(All_cnts_bub_in_paper[cau*4 + idx])
                    cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2),(0,0,255),3) # RED
                #	brow_img(paper,'paper')
            means = []
    #ket_qua_thi='pppppp'
    if len(questions_answer)>0:
        diem = float("{:.2f}".format(10*so_cau_dung/len(questions_answer)))
        #print(str(diem))
        #ket_qua_thi = 'Diem : '+str(10*round(so_cau_dung/len(questions_answer),3))+' (Ti le cau dung: '+str(so_cau_dung)+'/'+str(len(questions_answer))+')'
        ket_qua_thi = 'Diem : '+str(diem)+' (Ti le cau dung: '+str(so_cau_dung)+'/'+str(len(questions_answer))+')'
    else:
        ket_qua_thi = 'Sorry!!!'    
    #brow_img(paper,'paper')
    return ket_qua_thi , paper

def cham_ptn_001_40(image,dic_dap_an):
    paper = Find_4kv_va_scanhoa(image)
    paper = Xen_xquanh_anh(paper,beday=10)
    #brow_img(paper,'xxxxxx')
    cnts = Find_cnts_voi_kieuN(paper,kieuN=1)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    xM,yM,wM,hM = cv2.boundingRect(cnts[0]) #cnt Max chua trac nghiem
    anh_phan_tn = paper[yM:yM+hM,xM:xM+wM]
    anh_phan_tn = Xen_xquanh_anh(anh_phan_tn,beday=10)

    #cv2.drawContours(anh_phan_tn, cnts[:1], -1, (0, 0, 255), 5)
    #brow_img(anh_phan_tn,'anh_phan_tn')
    cnts = Find_cnts_voi_kieuN(anh_phan_tn,kieuN=0)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnt_bubs=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w >= 50 and (0.8 < float(w/h) < 1.2):
            cnt_bubs.append(c)

    #cv2.drawContours(anh_phan_tn, [c], 0, (0, 0,255), 4)
    #brow_img(anh_phan_tn,'XXXXXXXXXXXXX')
    #print(len(cnt_bubs)) #ra 160 thi dung

    if len(cnt_bubs) != 160:
        exit(print('Khong du so bubs'))
    #brow_img(anh_each_khoi,str(len(cnt_bubs)))
    cnt_bubs = sorted(cnt_bubs, key=get_x_ver0)
    cnts_all_bubs_sx=[]
    for i in range(4):
        khoi_cau = cnt_bubs[(i%4)*40 :(i%4)*40+40]
        khoi_cau = sorted(khoi_cau, key=get_y_ver1)
        for j in range(10):
            dong_cau=khoi_cau[(j%10)*4 :(j%10)*4+4]
            dong_cau = sorted(dong_cau, key=get_x_ver0)
            dong_cau_toadocu=[]
            for cc in dong_cau:
                dong_cau_toadocu.append(cc+np.array([xM+10,yM+10])) 
            cnts_all_bubs_sx = cnts_all_bubs_sx + dong_cau_toadocu
    #print(len(cnts_all_bubs_sx))
    #for c in cnts_all_bubs_sx:
    #    cv2.drawContours(paper, [c], 0, (0,0,255), 4)
    #    brow_img(paper,'XXX')
    #dic_dap_an = Tao_dicdapan_random(40)
    ########################################
    #ket_qua_thi , paper = Xu_li_bub_tinh_diem_thi(cnts_all_bubs_sx, image,dic_dap_an)
    #brow_img(paper,'XXX')
    #def Xu_li_bub_tinh_diem_thi(All_cnts_bub_in_paper, paper,dic_dap_an):
    All_cnts_bub_in_paper=cnts_all_bubs_sx
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
                cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2),(0,255,0),6) # GREEN #bk cong them 2 cho ro
                so_cau_dung = so_cau_dung+1
                #brow_img(paper,'paper')
            else:
                if answer_choices[min_arg]=='?':
                    # Lay idx cua dap an cau nay
                    kitu = dic_dap_an[cau]	# A hoac B hoac C hoac D
                    idx = answer_choices.index(kitu)
                    xb,yb,wb,hb = cv2.boundingRect(All_cnts_bub_in_paper[cau*4 + idx])
                    cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2),(225,0,225),6) # PRINK
                #	brow_img(paper,'paper'))
                else:
                    kitu = dic_dap_an[cau]	# A hoac B hoac C hoac D
                    idx = answer_choices.index(kitu)
                    xb,yb,wb,hb = cv2.boundingRect(All_cnts_bub_in_paper[cau*4 + idx])
                    cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2),(0,0,255),6) # RED
                #	brow_img(paper,'paper')
            means = []
    #ket_qua_thi='pppppp'
    if len(questions_answer)>0:
        diem = float("{:.2f}".format(10*so_cau_dung/len(questions_answer)))
        #print(str(diem))
        ket_qua_thi = 'Diem : '+str(10*round(so_cau_dung/len(questions_answer),3))+' (Ti le cau dung: '+str(so_cau_dung)+'/'+str(len(questions_answer))+')'
        #ket_qua_thi = 'Diem : '+str(diem)+' (Ti le cau dung: '+str(so_cau_dung)+'/'+str(len(questions_answer))+')'
        text1=ket_qua_thi
        #toado1 = (25,18)
        toado1 = (52,440)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color1 = (0, 0,255)
        thickness = 2
        str_sobd='??????'
        str_somd='???'
        cv2.putText(paper, 'So BD: '+str_sobd+', Ma De: '+str_somd+', '+text1, toado1, 0, fontScale, color1, thickness, cv2.LINE_AA)
        cv2.imwrite(str_sobd.replace('?','X')+'_'+str_somd.replace('?','X')+".jpg",paper)

    return paper
######################
#image=cv2.imread('ptn_001_40.jpg')
#dic_dap_an = Tao_dicdapan_random(40)

#paper = cham_ptn_001_40(image,dic_dap_an)
#brow_img(paper,'KQCC:')
