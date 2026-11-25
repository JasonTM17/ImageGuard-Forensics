# Thu nhận ảnh từ camera

Trong tab Phân tích hoặc Tăng cường, chọn **Chụp từ Camera…**, chọn nguồn rồi dùng **Xem thử thiết bị** hoặc **Chụp khung hình**. Nút dò cổng gợi ý index camera USB đang mở được.

Khi đang dò, xem thử hoặc chụp, các nút thao tác và lựa chọn nguồn được khóa đến khi hoàn tất. Hộp thoại tiếp tục phản hồi. **Hủy** đóng hộp thoại ngay. Lệnh đọc thiết bị hoặc mạng đang thực hiện có thể hoàn tất sau đó, và kết quả của lượt đã hủy được bỏ qua. Nếu người dùng chọn chụp lại hoặc hủy trước khi chấp nhận, dialog xoá file chụp chưa dùng và burst sidecar do chính thao tác đó tạo; file đã chấp nhận được giữ để phân tích. Nếu worker hoàn tất sau khi hủy, dialog cũng cố gắng xoá artifact của lượt chưa chấp nhận đó.

Lỗi thiết bị, ảnh xem trước hỏng hoặc lỗi chẩn đoán được hiển thị trong hộp thoại. Người dùng có thể đổi nguồn và thử lại sau khi thao tác kết thúc. Ảnh xem thử nằm trong thư mục tạm được dọn sau thao tác.

Luồng giao diện chụp lại lựa chọn nguồn khi bấm nút. Worker thực hiện I/O và chẩn đoán preview bằng QRunnable. Signal chuyển dữ liệu về luồng Qt để tạo QPixmap, cập nhật widget hoặc trả ảnh cho AnalyzeScreen và EnhanceScreen. Mỗi hộp thoại xử lý một thao tác tại một thời điểm.
