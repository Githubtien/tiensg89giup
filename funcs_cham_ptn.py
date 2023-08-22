import streamlit as st
from PIL import Image

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
