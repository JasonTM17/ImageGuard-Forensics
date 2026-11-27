"""Mọi text UI tập trung tại đây — tiếng Việt, bị test_integrity_strings quét.

Nguyên tắc wording: nói "dấu hiệu", không nói "kết luận"; KHÔNG dùng màu
đỏ/xanh tuyệt đối cho verdict (đỏ = phán có tội, xanh = minh oan — cả hai
đều overclaim cho công cụ phân tích tự động).
"""

APP_TITLE = "ImageGuard Forensics"

ENHANCE_BANNER = (
    "Ảnh đã tăng cường bằng thuật toán hoặc mô hình AI. Đây là bản xem, "
    "không phải bằng chứng. Phân tích dấu hiệu chỉ gắn với file gốc."
)

ENHANCE_AI_BANNER = (
    "Dạng xem này dùng mô hình AI; một số chi tiết có thể do mô hình tạo ra. "
    "Đây không phải bằng chứng; phân tích dấu hiệu vẫn dựa trên file gốc."
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
    "noise_local": "Noise local-MAD (vùng lạ cục bộ)",
    "frequency_analysis": "Frequency-domain cues",
    "splicing_cues": "Splicing cues (noise×ELA)",
    "deepfake_cues": "Dấu hiệu Deepfake / AI (biên khuôn mặt & phổ)",
    "deepfake_ml": "Deepfake ML — GenConViT (model học sâu)",
    "deepfake_screen_coc": "Deepfake qua màn hình — CoC Xception",
    "deepfake_ensemble": "Nhận diện Deepfake tổng hợp (ensemble nhiều mô hình)",
    "deepfake": "Dấu hiệu Deepfake / AI",
}

ENHANCE_OP_SUPER_RES = "Siêu phân giải cổ điển (Bicubic + IBP)"
ENHANCE_OP_CAM_CALIBRATE = "Hiệu chuẩn Cảm biến & Khớp màu Camera"
ENHANCE_OP_FREQ_SHARPEN = "Khử nhiễu & Làm nét Miền tần số (FFT)"
CAMERA_BTN_OPEN = "Chụp từ Camera…"
CAMERA_DIALOG_TITLE = "Thu nhận khung hình từ Camera"
CAMERA_SOURCE_WEBCAM = "Webcam máy tính (Thiết bị 0)"
CAMERA_SOURCE_USB = "Camera USB ngoài (Thiết bị 1..5)"
CAMERA_SOURCE_DUAL = "Chụp song song cả 2 Camera (Webcam 0 + Camera ngoài IoT/USB)"
CAMERA_SOURCE_IOT = "Camera IP / IoT ngoài (HTTP Snapshot URL)"
CAMERA_FORENSIC_RULE_NOTE = (
    "Nguyên tắc xử lý ảnh: Luôn đưa khung hình gốc từ cảm biến vào nhánh "
    "Phân tích (Forensics / Deepfake) trước; nhánh Khôi phục / Siêu phân giải "
    "chỉ dùng để hỗ trợ quan sát."
)
CAMERA_BTN_PROBE = "Quét cổng Camera"
A11Y_CAMERA_OPEN = "Chụp khung hình trực tiếp từ webcam hoặc camera ngoài"

BTN_DEMO_SAMPLE = "Tạo kịch bản ảnh mẫu…"
DEMO_DIALOG_TITLE = "Bộ tạo kịch bản kiểm thử Forensics & Deepfake"
A11Y_DEMO_SAMPLE = "Tạo ảnh mẫu tổng hợp để kiểm chứng phát hiện thao túng"

BTN_PRESET_DEFAULT = "Mặc định (3)"
BTN_PRESET_CAMERA = "Chế độ Camera / AI"
BTN_PRESET_ALL = "Tất cả"
BTN_DEEPFAKE_MODEL = "Nạp mô hình DIP…"
BTN_DEEPFAKE_MODEL_CLEAR = "Dùng trọng số mặc định"
DEEPFAKE_MODEL_DEFAULT = "Dấu hiệu vùng mặt: đang dùng trọng số heuristic mặc định."
DEEPFAKE_MODEL_SELECTED = "Mô hình đã học: {name}. Chỉ áp dụng khi chọn Dấu hiệu Deepfake / AI."
DEEPFAKE_MODEL_LOAD_ERROR = "Không dùng được mô hình: {reason}"
DEEPFAKE_MODEL_APPLIED = "Đang dùng trọng số học từ dữ liệu dev; chỉ số 0–1 chưa hiệu chuẩn xác suất."
DEEPFAKE_MODEL_UNAVAILABLE = "Đã chọn mô hình học từ dữ liệu nhưng khung này chưa đủ điều kiện áp dụng."
DEEPFAKE_MODEL_TRAINING_OVERLAP = "Ảnh này có trong dữ liệu phát triển; không dùng để báo cáo kết quả test."
BTN_SEND_TO_ENHANCE = "Đưa sang Tăng cường / Siêu phân giải →"

BTN_OPEN = "Mở ảnh…"
BTN_RUN = "Phân tích"
BTN_CANCEL = "Hủy"
BTN_CLOSE = "Đóng"
BTN_DELETE = "Xóa"
BTN_REOPEN = "Mở lại"
TAB_ANALYZE = "Phân tích"
TAB_ENHANCE = "Tăng cường"
TAB_HISTORY = "Lịch sử"
TAB_SETTINGS = "Cài đặt"

ERR_TITLE = "Không mở được ảnh"
FMT_HINT = "Định dạng hỗ trợ: JPEG, PNG, BMP, TIFF, WebP"
CONFIRM_DELETE = "Xóa phân tích này khỏi lịch sử (kèm artifacts)?"
EMPTY_HISTORY = "Chưa có phân tích nào."

TAB_STATS = "Thống kê"

LBL_METHODS = "Phương pháp:"
LBL_ENHANCE_METHODS = "Phương pháp tăng cường:"
LBL_PARAMS_GROUP = "Tham số"

BTN_ENHANCE_RUN = "Tăng cường"
BTN_EXPORT_REPORT = "Xuất báo cáo (Markdown)"
BTN_EXPORT_REPORT_SHORT = "Xuất báo cáo"
BTN_EXPORT_ENHANCED = "Xuất ảnh tăng cường (PNG)"
BTN_REFRESH = "Làm mới"

OVERLAY_NONE = "Không dùng lớp phủ"
FMT_OVERLAY_SKIP_DIMS = (
    "Overlay bỏ qua: kích thước {kind} {dims} không khớp kích thước ảnh {shape}"
)
OVERLAY_MASK_FMT = "Lớp phủ vùng — {label}"
OVERLAY_HEATMAP_FMT = "Bản đồ nhiệt — {label}"
OVERLAY_PAIRS_FMT = "Cặp điểm nối — {label}"
# Giữ tên cũ — trỏ sang format mới (placeholder {label}, không phải method id).
FMT_OVERLAY_MASK = OVERLAY_MASK_FMT
FMT_OVERLAY_HEATMAP = OVERLAY_HEATMAP_FMT
FMT_OVERLAY_TWIN = OVERLAY_PAIRS_FMT

STATUS_ENHANCE_CANCELLED = "Đã hủy tăng cường."
STATUS_ANALYZE_CANCELLING = "Đang dừng phân tích…"
STATUS_ANALYZE_CANCELLED = "Đã hủy phân tích."
STATUS_CLOSING_JOBS = "Đang dừng tác vụ trước khi đóng…"

HIST_COL_TIME = "Thời gian"
HIST_COL_IMAGE = "Ảnh"
HIST_COL_VERDICT = "Dấu hiệu"
HIST_COL_METHODS = "Số phương pháp"
HIST_COL_ID = "ID"

STATS_COL_VERDICT = "Dấu hiệu"
STATS_COL_COUNT = "Số lượng"
FMT_STATS_SUMMARY = "{n} phân tích · thời gian TB: {avg_ms:.0f} ms"

VERDICT_SIGNALS_PREFIX = "Dấu hiệu: "

LBL_BLEND_ORIGIN = "Gốc"
LBL_BLEND_ENHANCED = "Tăng cường"
FMT_BLEND_VALUE = "{value} %"

CANVAS_PLACEHOLDER = (
    "Kéo thả ảnh vào đây hoặc bấm 'Mở ảnh…' để bắt đầu"
)

MSG_HISTORY_NO_SELECTION = "Chọn một dòng trong bảng lịch sử trước."
MSG_BUSY_CANNOT_OPEN = "Đang xử lý — chưa thể mở ảnh khác."

ERR_SELECT_METHOD = "Vui lòng chọn ít nhất một phương pháp phân tích."
ERR_SOURCE_CHANGED_REOPEN = (
    "File ảnh đã thay đổi kể từ khi mở — mở lại ảnh để phân tích."
)

SETTINGS_THRESHOLDS_INTRO = (
    "Ngưỡng thuật toán (configs/thresholds.yaml) — mỗi giá trị kèm "
    "nguồn; chỉnh sửa bằng cách sửa file yaml rồi khởi động lại."
)
SETTINGS_TREE_HDR_KEY = "Khóa"
SETTINGS_TREE_HDR_VALUE = "Giá trị"
SETTINGS_TREE_HDR_SOURCE = "Nguồn"

HIST_ARTIFACT_PLACEHOLDER = "Chọn artifact…"
ARTIFACT_KIND_LABEL = {
    "mask": "Lớp phủ vùng",
    "heatmap": "Bản đồ nhiệt",
    "pairs": "Cặp điểm nối",
}
FMT_HIST_DETAIL_SUMMARY = (
    "{n} phương pháp · {ms} ms · cấu hình `{cfg}`"
)
MSG_EXPORT_OVERWRITE_IMAGE = "Ảnh đích đã tồn tại:\n{name}\nGhi đè?"
MSG_EXPORT_OVERWRITE_SIDECAR = (
    "Tệp siêu dữ liệu đi kèm đã tồn tại:\n{name}\nBạn có muốn ghi đè không?"
)
MSG_EXPORT_OVERWRITE_BOTH = (
    "Ảnh và tệp siêu dữ liệu đã tồn tại:\n{name}\n{sidecar}\nGhi đè?"
)
EMPTY_STATS = "Chưa có dữ liệu phân tích trong phiên này."

# --- Accessible names (screen readers) ---
A11Y_CANVAS = "Khung xem ảnh"
A11Y_CANVAS_DESC = (
    "Thu phóng bằng con lăn hoặc phím +/−; mũi tên để di chuyển; "
    "Home đặt lại khung nhìn; kéo để di chuyển; thả file ảnh để mở."
)

A11Y_ANALYZE_OPEN = "Mở ảnh để phân tích"
A11Y_ANALYZE_RUN = "Chạy phân tích dấu hiệu"
A11Y_ANALYZE_CANCEL = "Hủy phân tích đang chạy"
A11Y_ANALYZE_PROGRESS = "Tiến độ phân tích"
A11Y_ANALYZE_OVERLAY = "Chọn lớp phủ xem trên ảnh"
A11Y_ANALYZE_EXPORT = "Xuất báo cáo phân tích Markdown"
A11Y_ANALYZE_METHODS = "Chọn phương pháp phân tích"
FMT_A11Y_ANALYZE_METHOD = "{group} — {label}"

MSG_EXCEPTION_PATH_REDACTED = "Thao tác thất bại (chi tiết đường dẫn đã được ẩn)."
A11Y_ANALYZE_CARDS = "Danh sách kết quả theo phương pháp"

A11Y_ENHANCE_OPEN = "Mở ảnh để tăng cường"
A11Y_ENHANCE_OPERATOR = "Chọn toán tử tăng cường"
A11Y_ENHANCE_RUN = "Chạy tăng cường ảnh"
A11Y_ENHANCE_CANCEL = "Hủy tăng cường đang chạy"
A11Y_ENHANCE_PROGRESS = "Tiến độ tăng cường"
A11Y_ENHANCE_BLEND = "Trộn ảnh gốc và ảnh tăng cường"
A11Y_ENHANCE_BLEND_VALUE = "Mức trộn hiện tại"
A11Y_ENHANCE_EXPORT = "Xuất ảnh tăng cường PNG"
A11Y_ENHANCE_PARAMS = "Tham số toán tử tăng cường"

A11Y_HISTORY_TABLE = "Bảng lịch sử phân tích"
A11Y_HISTORY_REOPEN = "Mở lại phân tích đã chọn"
A11Y_HISTORY_EXPORT = "Xuất báo cáo phân tích đã chọn"
A11Y_HISTORY_DELETE = "Xóa phân tích đã chọn"
A11Y_HISTORY_STATUS = "Trạng thái thao tác lịch sử"

A11Y_STATS_TABLE = "Bảng thống kê theo dấu hiệu"
FMT_A11Y_METHOD_CARD = "Kết quả chi tiết phương pháp {title}"
A11Y_HIST_DETAIL_ARTIFACT = "Chọn artifact xem lại"
A11Y_STATS_REFRESH = "Làm mới thống kê phiên"

A11Y_SETTINGS_TREE = "Cây ngưỡng thuật toán"

A11Y_VERDICT_BANNER = "Tóm tắt dấu hiệu tổng thể"

A11Y_MAIN_TABS = "Điều hướng chức năng chính"

DEEPFAKE_PREVIEW_TITLE = "Dấu hiệu khám phá trên khung gốc"
DEEPFAKE_PREVIEW_DISCLAIMER = (
    "Đây là tín hiệu xử lí ảnh trên khung hình gốc, chưa có bộ phân loại Deepfake "
    "được huấn luyện. Kết quả không phải kết luận ảnh giả "
    "và không nhận diện danh tính."
)
DEEPFAKE_PREVIEW_SCORE = (
    "Điểm dấu hiệu: {score:.3f}/1 (chưa hiệu chuẩn; không phải xác suất thật/giả)."
)
DEEPFAKE_PREVIEW_NO_FRAMES = "Không có khung hình để phân tích; hãy chụp lại burst."
DEEPFAKE_CUE_FLAG_ON = "Cờ dấu hiệu không gian: bật; cần đối chiếu thêm."
DEEPFAKE_CUE_FLAG_OFF = "Cờ dấu hiệu không gian: chưa bật; không chứng minh ảnh thật."
DEEPFAKE_CUE_UNAVAILABLE = "Chưa đủ điều kiện đo dấu hiệu vùng mặt."
DEEPFAKE_REFERENCE_STATUS = {
    "difference": "Mốc phiên: có chênh lệch kết cấu cần đối chiếu thêm.",
    "no_difference_flag": "Mốc phiên: chênh lệch kết cấu dưới ngưỡng tham chiếu.",
    "same_frame": "Mốc phiên trùng khung đang phân tích; chưa có so sánh độc lập.",
    "face_unavailable": "Mốc phiên: chưa tìm được vùng mặt ở cả hai khung.",
    "quality_unavailable": "Mốc phiên: vùng mặt thiếu chất lượng để đối chiếu (kích thước, tương phản hoặc phơi sáng).",
    "comparison_unavailable": "Mốc phiên: cấu trúc hai vùng mặt chưa khớp để so sánh texture; hãy chụp cùng góc và khoảng cách.",
    "unavailable": "Mốc phiên chưa áp dụng vì phân tích vùng mặt không khả dụng.",
    "unsupported_capture_layout": "Mốc phiên chưa hỗ trợ so sánh ảnh ghép hai camera.",
}
CAMERA_BTN_ACCEPT_FRAME = "Dùng khung này để phân tích"
CAMERA_BTN_RETAKE = "Chụp lại"
ANALYZE_BTN_SET_REFERENCE = "Lưu khung này làm mốc cùng phiên"
ANALYZE_BTN_CLEAR_REFERENCE = "Xóa mốc cùng phiên"
ANALYZE_REFERENCE_READY = (
    "Đã có mốc cùng phiên trong 8 giờ. Lần phân tích sau sẽ so residual, LBP "
    "và FFT trong vùng mặt. Không nhận diện danh tính."
)
ANALYZE_REFERENCE_MISSING = "Chưa có mốc cùng phiên."
ANALYZE_REFERENCE_DUAL_UNSUPPORTED = (
    "Ảnh ghép hai camera chưa hỗ trợ mốc cùng phiên. "
    "Hãy chụp riêng từ một camera để dùng chức năng đối chiếu mốc."
)
ANALYZE_CAMERA_NEXT_STEP = (
    "Khung camera này không phải kết luận deepfake. "
    "Lưu nó làm mốc cùng phiên, rồi phân tích một khung nghi vấn khác."
)
CAMERA_BURST_CHECK = "Chụp 5 khung trong vài giây để xem mặt có đứng yên"
CAMERA_BURST_HINT = (
    "Độ tĩnh chỉ mô tả các khung đã chụp, không cộng vào điểm Deepfake. "
    "Người thật cũng có thể đứng yên; chuyển động cũng không chứng minh ảnh thật."
)
CAMERA_DUAL_HINT = (
    "Phát hiện thêm một camera. Có thể chụp song song để so hai cảm biến. "
    "Một webcam vẫn phân tích được."
)
CAMERA_SINGLE_LIMIT = (
    "Một camera: sáu trụ cột trên khung gốc, không so chéo cảm biến."
)
DEEPFAKE_PREVIEW_LIVE_TEXTURE = (
    "Tín hiệu làm mịn và biến đổi vi kết cấu ở vùng mặt đều thấp."
)
DEEPFAKE_VERDICT_SCREEN = "Không phải kết luận deepfake."
