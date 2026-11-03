"""Mọi text UI tập trung tại đây — tiếng Việt, bị test_integrity_strings quét.

Nguyên tắc wording: nói "dấu hiệu", không nói "kết luận"; KHÔNG dùng màu
đỏ/xanh tuyệt đối cho verdict (đỏ = phán có tội, xanh = minh oan — cả hai
đều overclaim cho công cụ phân tích tự động).
"""

APP_TITLE = "ImageGuard Forensics"

DISCLAIMER = (
    "Công cụ phát hiện dấu hiệu nghi ngờ — không kết luận ảnh bị can thiệp; "
    "ELA/EXIF/noise chỉ là bằng chứng khám phá."
)

VERDICT_LABEL = {
    "suspicious_indication": "Có dấu hiệu nghi vấn",
    "no_indication": "Không thấy dấu hiệu",
    "inconclusive": "Không đủ bằng chứng",
    "not_applicable": "Không áp dụng",
    "error": "Lỗi",
}

VERDICT_COLOR = {  # trung tính có chủ đích — không đỏ/xanh phán quyết
    "suspicious_indication": "#d97706",   # cam
    "no_indication": "#6b7280",           # xám
    "inconclusive": "#b7791f",            # vàng đất trung tính
    "not_applicable": "#9ca3af",
    "error": "#7c3aed",
}

METHOD_LABEL = {
    "copy_move_keypoint": "Copy-Move (keypoint SIFT/ORB)",
    "copy_move_block": "Copy-Move (block DCT)",
    "ela": "ELA (JPEG error levels)",
    "exif_audit": "Kiểm tra metadata EXIF",
    "noise_inconsistency": "Noise inconsistency",
    "frequency_analysis": "Frequency-domain cues",
    "splicing_cues": "Splicing cues (noise×ELA)",
}

BTN_OPEN = "Mở ảnh…"
BTN_RUN = "Phân tích"
BTN_CANCEL = "Hủy"
BTN_DELETE = "Xóa"
BTN_REOPEN = "Mở lại"
TAB_ANALYZE = "Phân tích"
TAB_HISTORY = "Lịch sử"
TAB_SETTINGS = "Cài đặt"

ERR_TITLE = "Không mở được ảnh"
FMT_HINT = "Định dạng hỗ trợ: JPEG, PNG, BMP, TIFF, WebP"
CONFIRM_DELETE = "Xóa phân tích này khỏi lịch sử (kèm artifacts)?"
EMPTY_HISTORY = "Chưa có phân tích nào."
