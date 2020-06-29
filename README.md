# AI2020
Đồ án môn học Trí Tuệ Nhân Tạo
TÌM HIỂU CÔNG CỤ AI VÀ ỨNG DỤNG

GVHD :  Nguyễn Đình Hiển


Nhóm SV:

Phan Đình Nguyên	16520850

Đảo Khả Phong	16520922

Nguyễn Hoàng Thắng	18521394

Trần Xuân Ánh	17520255

Phan Duy Nam	17520783




Hướng dẫn sử dụng:

Vào link https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/ để xem hướng dẫn cài vggface và mtcnn
Keras 2.2.5, Tensorflow 1.14, nếu cài phiên bản lớn hơn có thể sẽ chạy ko được.
Cài Flask, matplotlib
Vào link https://drive.google.com/uc?id=1kpxjaz3pIMrAhEjm7hJxcBsxKNhfl8t2&export=download tải, giải nén và copy tất cả trong thư mục gốc trong project, trong đó train là ảnh của folder train trong tập data vn_celeb và test là hình để thử.


Vào link https://drive.google.com/file/d/11TzloUotQHYGCJ5fknz9Pxj6U-hP8uaV/view?usp=sharing để tải file embeded_face_train_resnet50_vptree_new.pickle và coppy vào thư mục gốc của project. File này lưu vector đặc trưng của các ảnh trong folder faces.
Gõ python app.py trên terminal để khởi động ứng dụng, dùng trình duyệt vào link localhost:5000 để sử dụng ứng dụng.
