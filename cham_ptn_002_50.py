#cach tim 4 khoi vuong den to nhat de scan hoa du anh chup cach nao, mien la co 4 kv den to nhat
#B1: doi img ra anh xam roi sang anh nhi phan roi tim cac cnts cap TREE.
#B2: Duyet cac cnts tim duoc loc ra cac cnts co 4 canh va ti so w:h co 0.8-1.2
#B3: Sap xep cac cnts o B2 theo anh hcn bao quanh cnt voi np.mean cua no
#B4: Lay khoang 20 cai roi sx theo dien tich giam
#B5: Lay ra 4 cnts dau thi do la 4 KV den can tim, roi scan hoa
#sau khi scan hoa, lam lai nhu tren vi bay gio toado no khac. Ta se lay tiep 8 kv ke tiep
import numpy as np
import cv2
import os.path
import imutils
import random

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

def get_dosang_(s):
    # s = [thresh,cnt]
    xs,ys,ws,hs = cv2.boundingRect(s[0])
    anh = s[1][ys:ys+hs, xs:xs+ws]
    #gray = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    total = np.nonzero(anh)
    return total

def get_idx1(s):
    #s=[cnt,dosang]
    return s[1]

def get_val(s):
    # s=[cnt,val]
    return s[1]

def Find_20kv_den(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True )
    cnts_kv=[]
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and (0.8 < float(w)/h < 1.2):
            anhc=gray[y:y+h,x:x+w]
            val=int(np.mean(anhc))
            ttin=[cnt,val]
            cnts_kv.append(ttin)
    cnts_kv = sorted(cnts_kv, key = get_val)
    cnt_20kvden=[]
    for c in cnts_kv[:20]:
        cnt_20kvden.append(c[0])
    cnt_20kvden = sorted(cnt_20kvden,key=cv2.contourArea,reverse=True)
    return cnt_20kvden

def Rut_ra_4kv_forscan(cnt_20kvden):
    cnts_4kv=cnt_20kvden[:4]
    cnts_4kv = sorted(cnts_4kv, key= get_y_ver1)
    cnt_4c_TOP = cnts_4kv[:2]   # lay 2 cnt dau tien hy vng la 2 kv
    cnt_4c_BOT = cnts_4kv[-2:] # lay 2 cnt cuoi cung  hy vng la 2 kv
    cnt_4c_TOP = sorted(cnt_4c_TOP, key= get_x_ver0)
    cnt_4c_BOT = sorted(cnt_4c_BOT, key= get_x_ver0)
    paper = Scanhoa_from_4dinh_of4kv_mark(image,cnt_4c_TOP, cnt_4c_BOT)
    #brow_img(paper,'paper')
    return paper

def Find_cnts_voi_kieuN(image,kieuN):
    # kieuN = 0,1,2,3,4 ung voi cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP, cv2.ETR_TREE, cv2.RETR_FLOODFILL 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, kieuN, cv2.CHAIN_APPROX_SIMPLE)	
    cnts = imutils.grab_contours(cnts)
    return cnts

def Lay_so_ma_de(anh ,cnts_bulay_sx):
    str_somd=''
    means=[]
    gray = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY) 
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

def Lay_so_bao_danh(anh ,cnts_bulay_sx):
    str_sobd=''
    means=[]
    gray = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY) 
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
            str_sobd=str_sobd+kitulay
            means=[]
    #print(str_somd)
    return str_sobd            

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


def cham_ptn_002_50(image,dic_dap_an):
    cnt_20kvden = Find_20kv_den(image)
    paper = Rut_ra_4kv_forscan(cnt_20kvden)
    #brow_img(paper,'da scanhoa va xen')
    cnt_20kvden = Find_20kv_den(paper)
    cnt_8kvdenke=cnt_20kvden[4:12]
    #sx lai
    cnt_8kvdenke = sorted(cnt_8kvdenke,key=get_y_ver1)
    cnt_8kv_2d=cnt_8kvdenke[:2]
    cnt_8kv_2d = sorted(cnt_8kv_2d,key=get_x_ver0)
    cnt_8kv_3g=cnt_8kvdenke[2:5]
    cnt_8kv_3g = sorted(cnt_8kv_3g,key=get_x_ver0)
    cnt_8kv_3c=cnt_8kvdenke[5:]
    cnt_8kv_3c = sorted(cnt_8kv_3c,key=get_x_ver0)

    #for c in cnt_8kv_2d+cnt_8kv_3g+cnt_8kv_3c:
    #    cv2.drawContours(paper, [c], 0, (0, 0, 255), 5)
    #    brow_img(paper,'c')
    # lay ma de
    x0,y0,w0,h0 = cv2.boundingRect(cnt_8kv_2d[0])
    x3,y3,w3,h3 = cv2.boundingRect(cnt_8kv_3g[1])
    anh=paper[y0+h0:y3-int(0.1*(y3-y0)),x0+w0:x3-int(0.4*(x3-x0))]
    anh = Xen_xquanh_anh(anh,beday=4)
    cnts = Find_cnts_voi_kieuN(anh,kieuN=0)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #cv2.drawContours(anh, cnts, -1, (0, 0, 255), 5)
    #brow_img(anh,'c')
    cnts_bubs_md=cnts[:30]
    cnts_bubs_md = sorted(cnts_bubs_md,key=get_x_ver0)
    cnts_bulay_sx=[]
    for j in range(3):
        cnts_cot=cnts_bubs_md[(j%3)*10:(j%3)*10+10]
        cnts_cot=sorted(cnts_cot, key= get_y_ver1)
        cnts_bulay_sx=cnts_bulay_sx+cnts_cot
    #for c in cnts_bulay_sx:
    #    cv2.drawContours(anh, [c], -1, (0, 0, 255), 5)
    #    brow_img(anh,'c')
    str_somd = Lay_so_ma_de(anh ,cnts_bulay_sx)
    print(str_somd)
    
    # lay so bd
    x1,y1,w1,h1 = cv2.boundingRect(cnt_8kv_2d[1])
    x4,y4,w4,h4 = cv2.boundingRect(cnt_8kv_3g[2])
    anh=paper[y1+h1:y4-int(0.1*(y4-y1)),x1+w1:x4]
    anh = Xen_xquanh_anh(anh,beday=4)
    cnts = Find_cnts_voi_kieuN(anh,kieuN=0)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #cv2.drawContours(anh, cnts, -1, (0, 0, 255), 5)
    #brow_img(anh,'c')
    cnts_bubs_bd=cnts[:60]
    cnts_bubs_bd =sorted(cnts_bubs_bd,key=get_x_ver0) 
    cnts_bulay_sx=[]
    for j in range(6):
        cnts_cot=cnts_bubs_bd[(j%6)*10:(j%6)*10+10]
        cnts_cot=sorted(cnts_cot, key= get_y_ver1)
        cnts_bulay_sx=cnts_bulay_sx+cnts_cot
    #for c in cnts_bulay_sx:
    #    cv2.drawContours(anh, [c], -1, (0, 0, 255), 5)
    #    brow_img(anh,'c')
    str_sobd = Lay_so_bao_danh(anh ,cnts_bulay_sx)
    print(str_sobd)
    # trac nghiem 200 bubs 50 cau

    # lay khoi trac nghiem 2-6
    cnts_all_sx_tdcu=[]
    x2,y2,w2,h2 = cv2.boundingRect(cnt_8kv_3g[0])
    x6,y6,w6,h6 = cv2.boundingRect(cnt_8kv_3c[1])
    anh=paper[y2+h2:y6,x2+w2:x6]
    #anh = Xen_xquanh_anh(anh,beday=4)
    cnts = Find_cnts_voi_kieuN(anh,kieuN=0)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #cv2.drawContours(anh, cnts, -1, (0, 0, 255), 5)
    #brow_img(anh,'c')
    cnts_bubs_bd=cnts[:17*4]
    cnts_bubs_bd =sorted(cnts_bubs_bd,key=get_y_ver1) 
    cnts_bulay_sx=[]
    cnts_bulay_sx_tdcu=[]
    for j in range(17):
        cnts_cot=cnts_bubs_bd[(j%17)*4:(j%17)*4+4]
        cnts_cot=sorted(cnts_cot, key= get_x_ver0)
        cnts_bulay_sx=cnts_bulay_sx+cnts_cot
    for c in cnts_bulay_sx:
        cnts_bulay_sx_tdcu.append(c+np.array([x2+w2,y2+h2]))
    cnts_all_sx_tdcu=cnts_all_sx_tdcu+cnts_bulay_sx_tdcu

    #cv2.drawContours(paper, cnts_bulay_sx_tdcu, -1, (0, 0, 255), 5)
    #brow_img(paper,'c')

    # lay khoi trac nghiem 3-7
    x3,y3,w3,h3 = cv2.boundingRect(cnt_8kv_3g[1])
    x7,y7,w7,h7 = cv2.boundingRect(cnt_8kv_3c[2])
    anh=paper[y3+h3:y7,x3+w3:x7]
    #anh = Xen_xquanh_anh(anh,beday=4)
    cnts = Find_cnts_voi_kieuN(anh,kieuN=0)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #cv2.drawContours(anh, cnts, -1, (0, 0, 255), 5)
    #brow_img(anh,'c')
    cnts_bubs_bd=cnts[:17*4]
    cnts_bubs_bd =sorted(cnts_bubs_bd,key=get_y_ver1) 
    cnts_bulay_sx=[]
    cnts_bulay_sx_tdcu=[]
    for j in range(17):
        cnts_cot=cnts_bubs_bd[(j%17)*4:(j%17)*4+4]
        cnts_cot=sorted(cnts_cot, key= get_x_ver0)
        cnts_bulay_sx=cnts_bulay_sx+cnts_cot
    for c in cnts_bulay_sx:
        cnts_bulay_sx_tdcu.append(c+np.array([x3+w3,y3+h3]))
    cnts_all_sx_tdcu=cnts_all_sx_tdcu+cnts_bulay_sx_tdcu
    #cv2.drawContours(paper, cnts_bulay_sx_tdcu, -1, (0, 0, 255), 5)
    #brow_img(paper,'c')

    # lay khoi trac nghiem 4-8
    x4,y4,w4,h4 = cv2.boundingRect(cnt_8kv_3g[2])
    x8,y8,w8,h8 = x7+int(0.7*(x7-x6)),y7,w7,h7
    anh=paper[y4+h4:y8,x4+w4:x8]
    #anh = Xen_xquanh_anh(anh,beday=4)
    cnts = Find_cnts_voi_kieuN(anh,kieuN=0)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #cv2.drawContours(anh, cnts, -1, (0, 0, 255), 5)
    #brow_img(anh,'c')
    cnts_bubs_bd=cnts[:16*4]
    cnts_bubs_bd =sorted(cnts_bubs_bd,key=get_y_ver1) 
    cnts_bulay_sx=[]
    cnts_bulay_sx_tdcu=[]
    for j in range(16):
        cnts_cot=cnts_bubs_bd[(j%16)*4:(j%16)*4+4]
        cnts_cot=sorted(cnts_cot, key= get_x_ver0)
        cnts_bulay_sx=cnts_bulay_sx+cnts_cot
    for c in cnts_bulay_sx:
        cnts_bulay_sx_tdcu.append(c+np.array([x4+w4,y4+h4]))
    cnts_all_sx_tdcu=cnts_all_sx_tdcu+cnts_bulay_sx_tdcu
    #cv2.drawContours(paper, cnts_bulay_sx_tdcu, -1, (0, 0, 255), 5)
    #brow_img(paper,'c')
    #for c in cnts_all_sx_tdcu:
    #    cv2.drawContours(paper, [c], 0, (0, 0, 255), 5)
    #    brow_img(paper,'c')
    print(len(cnts_all_sx_tdcu))    
    #dic_dap_an = Tao_dicdapan_random(40)
    ########################################
    All_cnts_bub_in_paper=cnts_all_sx_tdcu[:4*len(dic_dap_an)]
    Xu_li_bub_tinh_diem_thi(All_cnts_bub_in_paper, paper,dic_dap_an)

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
        means.append(np.mean(anh))
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
                cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2),(0,255,0),3) # GREEN #bk cong them 2 cho ro
                so_cau_dung = so_cau_dung+1
                #brow_img(paper,'paper')
            else:
                if answer_choices[min_arg]=='?':
                    # Lay idx cua dap an cau nay
                    kitu = dic_dap_an[cau]	# A hoac B hoac C hoac D
                    idx = answer_choices.index(kitu)
                    xb,yb,wb,hb = cv2.boundingRect(All_cnts_bub_in_paper[cau*4 + idx])
                    cv2.circle(paper,(xb+round(wb/2),yb+round(hb/2)),round(wb/2),(225,0,225),3) # PRINK
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
        #diem = float("{:.2f}".format(10*so_cau_dung/len(questions_answer)))
        diem = round(10*so_cau_dung/len(questions_answer),1)
        #print(diem)
        #print(str(diem))
        ket_qua_thi = 'Diem : '+str(diem)+' (Ti le cau dung: '+str(so_cau_dung)+'/'+str(len(questions_answer))+')'
        #ket_qua_thi = 'Diem : '+str(diem)+' (Ti le cau dung: '+str(so_cau_dung)+'/'+str(len(questions_answer))+')'
        text1=ket_qua_thi
        #toado1 = (25,18)
        toado1 = (110,800)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        color1 = (0, 0,255)
        color2 = (255, 0,0)
        thickness = 3
        cv2.putText(paper, 'Ma de: ', (20,400), 0, fontScale, color2, thickness, cv2.LINE_AA)
        cv2.putText(paper, str_somd, (20,500), 0, fontScale, color2, thickness, cv2.LINE_AA)

        cv2.putText(paper, 'So bd: ', (690,400), 0, fontScale, color2, thickness, cv2.LINE_AA)
        cv2.putText(paper, str_sobd, (690,500), 0, fontScale, color2, thickness, cv2.LINE_AA)

        cv2.putText(paper, 'Diem : '+str(10*round(so_cau_dung/len(questions_answer),3)), (20,1400), 0, fontScale, color1, thickness, cv2.LINE_AA)
        cv2.putText(paper, ' (Ti le cau dung: '+str(so_cau_dung)+'/'+str(len(questions_answer)), (680,1400), 0, fontScale, color1, thickness, cv2.LINE_AA)

        #cv2.putText(paper, 'So BD: '+str_sobd+', Ma De: '+str_somd+', '+text1, toado1, 0, fontScale, color1, thickness, cv2.LINE_AA)
        #cv2.putText(paper, text1 + ', So BD: '+str_sobd+', Ma De: '+str_somd, toado1, 0, fontScale, color1, thickness, cv2.LINE_AA)
        #cv2.imwrite(str_sobd.replace('?','X')+'_'+str_somd.replace('?','X')+".jpg",paper)

    return paper

######################
#tep='PTN_HS/ptn-vh-001.jpg'
#if not os.path.exists(tep):
#    exit('Khong co tep : '+tep)

#image=cv2.imread(tep)
#image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#image = Xen_xquanh_anh(image,beday=40)
#brow_img(image,'X')

#dic_dap_an = Tao_dicdapan_random(socau=50)
#paper = cham_ptn_002_50(image,dic_dap_an)
#brow_img(paper,'KQCC:')
