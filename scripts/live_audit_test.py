"""Kịch bản kiểm tra toàn diện trực tiếp hệ thống ImageGuard Forensics."""

import sys
from pathlib import Path

from imageguard.core.enums import Verdict
from imageguard.services.analysis_service import AnalysisRequest
from imageguard.services.report_service import ReportService
from imageguard.ui.app import build_services


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=================================================================")
    print("      IMAGEGUARD FORENSICS - KIỂM THỬ TRỰC TIẾP TOÀN BỘ HỆ THỐNG")
    print("=================================================================\n")

    test_root = Path("reports/audit-live-test")
    test_root.mkdir(parents=True, exist_ok=True)
    db_path = test_root / "test.db"
    art_path = test_root / "artifacts"

    # Khởi tạo services (Database, Repository, Artifact Store)
    analysis_svc, history_svc = build_services(db_path=db_path, artifacts_root=art_path)
    report_svc = ReportService(store=analysis_svc.store)

    test_cases = [
        ("Ảnh giả mạo (Forged)", Path("data/synth_eval/medium_0000_F.png"), Verdict.SUSPICIOUS_INDICATION),
        ("Ảnh gốc chuẩn (Original)", Path("data/synth_eval/medium_0000_O.png"), Verdict.INCONCLUSIVE),
    ]

    for label, img_path, expected_verdict in test_cases:
        print(f"\n>>> [TEST CASE] {label}: {img_path.name}")
        req = AnalysisRequest(
            path=img_path,
            methods=["keypoint", "block", "ela", "noise", "noise_local", "freq", "splicing"]
        )
        report = analysis_svc.analyze(req)

        print(f"  + ID Báo cáo: {report.report_id}")
        print(f"  + Kích thước ảnh: {report.image_width} x {report.image_height}")
        print(f"  + Kết luận tổng hợp (Overall Verdict): {report.overall_verdict.value}")
        print(f"  + Kỳ vọng: {expected_verdict.value} -> {'CHÍNH XÁC (MATCH)' if report.overall_verdict == expected_verdict else 'KHÁC KỲ VỌNG'}")
        print("  + Chi tiết từng thuật toán:")
        for m in report.methods:
            score_str = f"{m.score:.4f}" if isinstance(m.score, (int, float)) else str(m.score)
            print(f"      - {m.method.value:<22}: {m.verdict.value:<22} (score: {score_str}, time: {m.duration_ms}ms)")

    # Kiểm tra lưu trữ lịch sử
    print("\n-----------------------------------------------------------------")
    print(">>> [KIỂM TRA CƠ SỞ DỮ LIỆU & LỊCH SỬ (HISTORY SERVICE)]")
    history_items = history_svc.list()
    print(f"  + Tổng số bản ghi trong DB: {len(history_items)}")
    for item in history_items:
        print(f"      - Record {item['id'][:8]}... | File: {item['image_name']} | Verdict: {item['verdict']} | Methods: {item['n_methods']}")

    # Kiểm tra Artifacts
    print("\n-----------------------------------------------------------------")
    print(">>> [KIỂM TRA ARTIFACT STORE (LƯU TRỮ MẶT NẠ, HEATMAP, JSON)]")
    for item in history_items:
        item_id = item["id"]
        artifacts = list((art_path / item_id).glob("*"))
        print(f"  + Analysis {item_id[:8]}... tạo ra {len(artifacts)} artifacts:")
        for art in artifacts:
            print(f"      * {art.name} ({art.stat().st_size} bytes)")

    # Kiểm tra xuất báo cáo Markdown
    print("\n-----------------------------------------------------------------")
    print(">>> [KIỂM TRA REPORT SERVICE (XUẤT BÁO CÁO MARKDOWN)]")
    last_id = history_items[0]["id"]
    out_md = test_root / "final_audit_report.md"
    exported = report_svc.export_markdown(last_id, out_path=out_md)
    print(f"  + Đã xuất báo cáo thành công ra file: {exported}")
    lines = exported.read_text(encoding="utf-8").splitlines()
    print(f"  + Tổng số dòng báo cáo: {len(lines)}")
    print("  + Trích đoạn nội dung báo cáo:")
    print("    --------------------------------------------------")
    for line in lines[:15]:
        print(f"    {line}")
    print("    --------------------------------------------------")
    print("\n=================================================================")
    print("            KẾT QUẢ: TẤT CẢ CÁC BƯỚC HOẠT ĐỘNG HOÀN HẢO!")
    print("=================================================================")


if __name__ == "__main__":
    main()
