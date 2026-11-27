"""Machine-learning forensic methods (ONNX, CPU, offline).

Khác ``forensics.auxiliary``: method trong package này là model học sâu đã
train trên dữ liệu thật (DFDC/FaceForensics++/Celeb-DF) nên có thể trả
``SUSPICIOUS_INDICATION`` khi score vượt ngưỡng đã hiệu chuẩn — vẫn là *dấu
hiệu*, không phải phán quyết sự thật. Mọi model là tùy chọn, pin bằng
SHA-256, không bao giờ tải lúc runtime và không commit vào repo.
"""
