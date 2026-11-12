"""Thuật toán hình thái học nhị phân (Chương 4 — Hình thái học, slides PGS.TS. Hoàng Văn Dũng).

Binary morphological algorithms from set theory: hit-or-miss, boundary, region filling,
connected components, convex hull, thinning, thickening, skeleton (Lantuejoul).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from imageguard.ipkit.morphology import _check_se

Connectivity = Literal[4, 8]


def require_binary_image(img: np.ndarray, *, name: str = "binary") -> np.ndarray:
    """Chuẩn hoá ảnh nhị phân: bool với True = tiền cảnh (foreground).

    Parameters
    ----------
    img
        Mảng 2D ``bool`` (0/1) hoặc ``uint8`` chỉ chứa 0 và 255.

    Returns
    -------
    np.ndarray
        Mảng ``bool`` 2D cùng kích thước; True tương ứng pixel đen/tiền cảnh trong slide.

    Raises
    ------
    ValueError
        Sai hình dạng, rỗng, hoặc giá trị không phải nhị phân.
    TypeError
        Kiểu dữ liệu không được hỗ trợ.

    Examples
    --------
    >>> require_binary_image(np.array([[0, 255], [255, 0]], dtype=np.uint8)).shape
    (2, 2)

    Ghi chú bước (tối thiểu hoá dùng bộ nhớ khi đã đúng kiểu)::

        uint8:  0→False, 255→True   (ví dụ 1×2: [0,255] → [F,T])
    """
    # Đưa đầu vào về ndarray để kiểm tra hình dạng/thuộc tính thống nhất (tiền xử lý morphology)
    arr = np.asarray(img)
    # Morphology nhị phân chỉ áp dụng trên lưới 2D có ít nhất một pixel
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    # bool/True=tiền cảnh: không copy nếu đã bool — tránh đổi hành vi tham chiếu không cần thiết
    if arr.dtype == np.bool_ or arr.dtype == bool:
        return arr.astype(bool, copy=False)
    if arr.dtype == np.uint8:
        # Chỉ 0 và 255: mọi giá trị xám khác không phải mask nhị phân chuẩn trong bài
        if not np.all((arr == 0) | (arr == 255)):
            raise ValueError(f"{name} uint8 must contain only 0 and 255")
        # 255≠0 trong uint8 nên astype(bool) đúng quy ước True=đối tượng đen/tiền cảnh
        return arr.astype(bool)
    # Các kiểu khác (float, int...) không được dùng trực tiếp cho phép toán tập hợp C4
    raise TypeError(f"{name} must be bool or uint8 (0/255)")


def _check_hit_or_miss_pair(
    se_fg: np.ndarray, se_bg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kiểm tra cặp SE hit-or-miss cùng kích thước lẻ, chỉ 0/1, không trùng foreground/background."""
    # B₁: tập vị trí trong SE phải khớp tiền cảnh khi erosion trên A (hit-or-miss / morphology)
    fg = _check_se(se_fg).astype(bool)
    # B₂: tập vị trí phải khớp nền khi erosion trên phần bù Aᶜ
    bg = _check_se(se_bg).astype(bool)
    # Hai mẫu cùng “cửa sổ” quanh gốc — lệch shape sẽ làm lệch tọa độ hit
    if fg.shape != bg.shape:
        raise ValueError("se_fg and se_bg must have the same shape")
    # Trên slide, B₁ và B₂ có thể cùng đánh dấu 1 tại một ô: lần lượt kiểm tra trên A và Aᶜ
    return fg, bg


def _binary_erode(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Co nhị phân: điểm z thuộc A⊖B khi mọi pixel của B dịch tới z nằm trong A (C4, erosion).

    Ví dụ tư duy 3×3, B toàn 1: pixel trung tâm chỉ giữ True nếu cả vùng 3×3 quanh nó
    trên A đều True — giống “lọc AND” theo structuring element.
    """
    elem = _check_se(se).astype(bool)
    h, w = mask.shape
    kh, kw = elem.shape
    # Tâm SE tại (ph,pw): padding bằng nửa kích thước để dịch B khớp mọi z trong ảnh gốc
    ph, pw = kh // 2, kw // 2
    # Nền ảo False: dịch B ra ngoài biên không thể “nằm gọn” trong A
    pad = np.pad(mask, ((ph, ph), (pw, pw)), mode="constant", constant_values=False)
    # Chỉ xét ô B=1 — ô 0 trong SE không ràng buộc (chuẩn morphology nhị phân)
    active = np.argwhere(elem)
    if active.size == 0:
        # SE rỗng / không có pixel bật: giao rỗng → co ra toàn False
        return np.zeros((h, w), dtype=bool)
    # Khởi tạo “mọi z đủ điều kiện”, rồi AND dần từng mẫu dịch B
    out = np.ones((h, w), dtype=bool)
    for ky, kx in active:
        # Cắt cửa sổ h×w của mask đã pad: tương ứng mask dịch theo offset (ky,kx) trong SE
        slab = pad[ky : ky + h, kx : kx + w]
        out &= slab
    return out


def _binary_dilate(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Giãn nhị phân: A⊕B là tập dịch z sao cho B phản xạ và A giao nhau (C4, dilation).

    Đối lập erosion: pixel z True nếu có **ít nhất một** dịch B (tại ô active của SE)
    sao cho vùng tương ứng trên mask có giao với True — tư duy “lọc OR”.
    """
    elem = _check_se(se).astype(bool)
    h, w = mask.shape
    kh, kw = elem.shape
    ph, pw = kh // 2, kw // 2
    # Cùng quy ước padding như erosion: biên ảnh coi là nền khi dịch SE
    pad = np.pad(mask, ((ph, ph), (pw, pw)), mode="constant", constant_values=False)
    active = np.argwhere(elem)
    if active.size == 0:
        return np.zeros((h, w), dtype=bool)
    # Bắt đầu từ rỗng, cộng (OR) các bản sao mask dịch theo từng điểm của B
    out = np.zeros((h, w), dtype=bool)
    for ky, kx in active:
        slab = pad[ky : ky + h, kx : kx + w]
        out |= slab
    return out


def _binary_open(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Mở nhị phân: giãn(co(A,B),B) — dùng cho phần thân xương Lantuejoul.

    Mở cắt gai nhỏ và làm phẳng biên cục bộ; trong skeleton, Aₖ−(Aₖ∘B) giữ “lõi”
    không bị mở nuốt mất.
    """
    # ∘ = mở: co trước (bỏ chi tiết nhỏ hơn B) rồi giãn (khôi phục kích thước vùng còn)
    return _binary_dilate(_binary_erode(mask, se), se)


def hit_or_miss(binary: np.ndarray, se_fg: np.ndarray, se_bg: np.ndarray) -> np.ndarray:
    """Biến đổi hit-or-miss A⊛B = (A⊖B₁) ∩ (Aᶜ⊖B₂) (C4 § Hit-or-Miss).

    Parameters
    ----------
    binary
        Ảnh nhị phân A.
    se_fg
        B₁: vị trí bắt buộc khớp tiền cảnh.
    se_bg
        B₂: vị trí bắt buộc khớp nền (trên Aᶜ).

    Returns
    -------
    np.ndarray
        ``bool``: True tại gốc SE khớp hoàn toàn mẫu.

    Examples
    --------
    >>> se = np.ones((3, 3), dtype=np.uint8)
    >>> fg = np.zeros((3, 3), dtype=np.uint8); fg[1, 1] = 1
    >>> bg = se - fg
    >>> img = np.zeros((5, 5), dtype=np.uint8); img[2, 2] = 255
    >>> hit_or_miss(img, fg, bg)[2, 2]
    np.True_
    """
    mask = require_binary_image(binary)
    fg, bg = _check_hit_or_miss_pair(se_fg, se_bg)
    # Erosion tiền cảnh: B₁ phải nằm trong A
    in_fg = _binary_erode(mask, fg.astype(np.uint8))
    # Erosion trên nền: B₂ phải nằm trong phần bù A
    in_bg = _binary_erode(~mask, bg.astype(np.uint8))
    # Hit tại z khi cả hai điều kiện erosion đồng thời đúng — giao tập hợp trên slide
    return in_fg & in_bg


def boundary_extraction(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Trích biên β(A) = A − (A ⊖ B) (C4 § Boundary Extraction).

    Parameters
    ----------
    binary
        Tập tiền cảnh A.
    se
        Phần tử cấu trúc B (kích thước lẻ); độ dày biên phụ thuộc B.

    Returns
    -------
    np.ndarray
        ``bool``: vành biên của A.

    Examples
    --------
    >>> sq = np.zeros((5, 5), dtype=np.uint8); sq[1:4, 1:4] = 255
    >>> b = boundary_extraction(sq, np.ones((3, 3), dtype=np.uint8))
    >>> int(b.sum())
    8
    """
    mask = require_binary_image(binary)
    elem = _check_se(se)
    # Co A bằng B: lõi “còn sau erosion” — pixel nằm sâu trong đối tượng
    eroded = _binary_erode(mask, elem)
    # Hiệu tập hợp: giữ pixel thuộc A nhưng bị mất sau co
    return mask & ~eroded


def region_filling(
    binary: np.ndarray,
    se: np.ndarray,
    *,
    seed: np.ndarray | None = None,
) -> np.ndarray:
    """Lấp vùng nền bằng lặp Xₖ = (Xₖ₋₁ ⊕ B) ∧ Aᶜ (C4 § Region Filling).

    Nếu ``seed`` là None: khởi tạo X₀ là mọi pixel nền (False trong A) **nối với biên ảnh**
    (nền “bên ngoài”), rồi giãn trong Aᶜ cho tới hội tụ; kết quả = A ∨ Xₖ (lấp các lỗ bên trong).

    Nếu ``seed`` cho trước (một điểm bool hoặc mask bool cùng kích thước): X₀ = seed ∧ Aᶜ,
    thích hợp khi đã biết một điểm bên trong lỗ.

    Parameters
    ----------
    binary
        Ảnh A; True = tiền cảnh.
    se
        B cho giãn (thường vuông 3×3).
    seed
        Tùy chọn: điểm/mask khởi tạo trong vùng cần lấp.

    Returns
    -------
    np.ndarray
        ``bool``: A đã được lấp các lỗ nội bộ (theo điều kiện trên).

    Examples
    --------
    >>> ring = np.full((7, 7), 255, dtype=np.uint8); ring[2:5, 2:5] = 0; ring[3, 3] = 255
    >>> filled = region_filling(ring, np.ones((3, 3), dtype=np.uint8))
    >>> filled[3, 2]
    np.True_
    """
    mask = require_binary_image(binary)
    elem = _check_se(se)
    # Aᶜ: vùng nền — chỉ giãn/lấp trong phần bù của tiền cảnh (region filling / morphology)
    ac = ~mask
    if seed is None:
        # Nền nối biên ảnh; phần nền còn lại là lỗ kín — hợp với A để lấp lỗ (auto border seed)
        exterior = _border_connected_background(ac)
        # Lỗ = nền không 4-liên thông ra viền ảnh (hố kín trong đối tượng)
        holes = ac & ~exterior
        return mask | holes
    s = np.asarray(seed)
    if s.dtype != np.bool_ and s.dtype != bool:
        raise TypeError("seed must be a bool array")
    if s.shape != mask.shape:
        raise ValueError("seed must be a bool mask with the same shape as binary")
    # Hạt ban đầu phải nằm trong lỗ (nền), không trên tiền cảnh
    x = s & ac
    prev = np.zeros_like(x)
    # Lặp giãn trong Aᶜ từ hạt bên trong như công thức Xₖ trên slide
    while not np.array_equal(x, prev):
        prev = x.copy()
        # Giãn nhưng cắt bởi Aᶜ — không “tràn” vào tiền cảnh
        x = _binary_dilate(x, elem) & ac
    # Hợp lỗ đã lan với A: lỗ nội bộ chuyển thành foreground
    return mask | x


def fill_holes(
    binary: np.ndarray,
    se: np.ndarray,
    *,
    seed: np.ndarray | None = None,
) -> np.ndarray:
    """Bí danh của :func:`region_filling` (lấp lỗ theo C4)."""
    return region_filling(binary, se, seed=seed)


def _border_connected_background(ac: np.ndarray) -> np.ndarray:
    """Đánh dấu mọi pixel nền (True trong ac) 4-liên thông với viền ảnh."""
    h, w = ac.shape
    # B vuông 3×3 ≈ giãn 8-neighborhood dùng trong lặp lan truyền morphology
    se = np.ones((3, 3), dtype=np.uint8)
    x = np.zeros((h, w), dtype=bool)
    # X₀: mọi pixel nền trên bốn cạnh ảnh — “cầu nối” ra vô cực ngoài khung
    x[0, :] |= ac[0, :]
    x[-1, :] |= ac[-1, :]
    x[:, 0] |= ac[:, 0]
    x[:, -1] |= ac[:, -1]
    prev = np.zeros_like(x)
    # Lặp giãn∩nền cho tới bão hoà: connected components / region filling phụ trợ
    while not np.array_equal(x, prev):
        prev = x.copy()
        x = (_binary_dilate(x, se) & ac) | x
    return x


def connected_components(
    binary: np.ndarray,
    connectivity: Connectivity = 4,
) -> tuple[np.ndarray, int]:
    """Gán nhãn thành phần liên thông (C4 § Extraction of Connected Components).

    Dùng **hai lượt union-find** (equivalence labels): O(HW α(n)), ổn định và dễ đếm
    số vùng; thay cho lặp giãn−giao từng hạt như minh hoạ slide khi cần mọi nhãn cùng lúc.

    Parameters
    ----------
    binary
        Ảnh nhị phân.
    connectivity
        4 hoặc 8 lân cận.

    Returns
    -------
    labels
        ``int32`` 2D; 0 = nền, 1..count = các thành phần.
    count
        Số thành phần foreground.

    Examples
    --------
    >>> a = np.zeros((3, 3), dtype=np.uint8); a[0, 0] = 255; a[2, 2] = 255
    >>> lbl, n = connected_components(a, 4)
    >>> n
    2
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    mask = require_binary_image(binary)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    # Quét một phần láng giềng đủ để không bỏ sót cạnh khi duyệt trái→phải, trên→dưới
    if connectivity == 4:
        offsets = [(-1, 0), (0, -1)]
    else:
        offsets = [(-1, 0), (0, -1), (-1, -1), (-1, 1)]
    next_label = 1
    parent: list[int] = [0]

    def find(i: int) -> int:
        # Path compression: rút ngắn cây nhãn tương đương (segmentation / components)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Lượt 1: gán nhãn tạm và ghi nhận tương đương với láng giềng đã quét
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            neigh_labels: list[int] = []
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] > 0:
                    neigh_labels.append(labels[ny, nx])
            if not neigh_labels:
                # Thành phần mới: tách nhãn chưa gộp
                parent.append(next_label)
                labels[y, x] = next_label
                next_label += 1
            else:
                # Gộp nhãn láng giềng: dùng min làm đại diện tạm (equivalence trên slide)
                m = min(neigh_labels)
                labels[y, x] = m
                for other in neigh_labels:
                    if other != m:
                        union(m, other)
    if next_label == 1:
        return labels, 0
    # Lượt 2: gom nhãn về gốc union-find
    remap: dict[int, int] = {}
    new_id = 0
    for y in range(h):
        for x in range(w):
            lab = labels[y, x]
            if lab == 0:
                continue
            root = find(lab)
            if root not in remap:
                new_id += 1
                remap[root] = new_id
            labels[y, x] = remap[root]
    return labels, new_id


def _convex_hull_structuring_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    """Bốn cặp SE hit-or-miss trong slide convex hull (hướng Đông, Bắc, Tây, Nam)."""
    # Mỗi hàng: (B₁,B₂) bắt “góc lõm” cục bộ để bù pixel làm lồi — morphology / hit-or-miss
    return [
        (
            np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.uint8),
            np.array([[0, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=np.uint8),
        ),
        (
            np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.uint8),
            np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=np.uint8),
        ),
        (
            np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint8),
            np.array([[0, 0, 1], [0, 0, 1], [0, 1, 1]], dtype=np.uint8),
        ),
        (
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8),
            np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.uint8),
        ),
    ]


def convex_hull(binary: np.ndarray) -> np.ndarray:
    """Bao lồi CH(A) lặp hit-or-miss bốn hướng rồi hợp với A (C4 § Convex Hull).

    Giới hạn đã biết: thuật toán slide chỉ mở rộng trong **hộp giới hạn** (bounding box)
    của A, không “bù” pixel ngoài khung bao quanh ảnh.

    Parameters
    ----------
    binary
        Tập A.

    Returns
    -------
    np.ndarray
        ``bool`` xấp xỉ bao lồi trong khung giới hạn.

    Examples
    --------
    >>> img = np.zeros((9, 9), dtype=np.uint8); img[4, 2:7] = 255; img[2:7, 4] = 255
    >>> hull = convex_hull(img)
    >>> hull[0, 0]
    np.False_
    """
    mask = require_binary_image(binary)
    work = mask.copy()
    pairs = _convex_hull_structuring_pairs()
    changed = True
    # Lặp đủ bốn mẫu cho tới khi không còn pixel mới
    while changed:
        changed = False
        for se_fg, se_bg in pairs:
            hits = hit_or_miss(work, se_fg, se_bg)
            if np.any(hits):
                # Hợp pixel hit vào tập làm việc — mở rộng dần về phía lõm
                work |= hits
                changed = True
    return work


def _thinning_structuring_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    """Bộ SE xoay 90° cho làm mảnh (hit-or-miss), theo chuỗi B₁…Bₙ trong slide."""
    # Mẫu gốc 3×3: bắt điểm đầu/cạnh có thể gỡ mà không đứt đối tượng (thinning hit-or-miss)
    base_fg = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    base_bg = np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    fg, bg = base_fg, base_bg
    # Xoay 90° bốn lần → bộ SE đối xứng quay cho mọi hướng biên (thinning morphology)
    for _ in range(4):
        pairs.append((fg.copy(), bg.copy()))
        fg = np.rot90(fg)
        bg = np.rot90(bg)
    return pairs


def thinning(binary: np.ndarray, *, max_passes: int = 64) -> np.ndarray:
    """Làm mảnh A ⊗ {Bᵢ}: lặp A ← A − (A ⊛ Bᵢ) tới khi ổn định (C4 § Thinning).

    Parameters
    ----------
    binary
        Ảnh cần mảnh.
    max_passes
        Giới hạn vòng lặp ngoài (an toàn cho ảnh lớn trong demo).

    Returns
    -------
    np.ndarray
        ``bool`` sau làm mảnh.

    Examples
    --------
    >>> thick = np.zeros((3, 7), dtype=np.uint8); thick[1, :] = 255
    >>> thin_once = thinning(thick)
    >>> thinning(thin_once).sum() == thin_once.sum()
    np.True_
    """
    if isinstance(max_passes, bool) or not isinstance(max_passes, (int, np.integer)):
        raise TypeError("max_passes must be an integer")
    if int(max_passes) < 1:
        raise ValueError("max_passes must be >= 1")
    work = require_binary_image(binary)
    pairs = _thinning_structuring_pairs()
    for _ in range(int(max_passes)):
        before = work.copy()
        for se_fg, se_bg in pairs:
            hits = hit_or_miss(work, se_fg, se_bg)
            # Hiệu tập hợp: bỏ các pixel khớp mẫu làm mảnh
            work &= ~hits
        # Dừng sớm khi một vòng cả bốn SE không đổi A — tiết kiệm vòng lặp demo
        if np.array_equal(work, before):
            break
    return work


def thickening(binary: np.ndarray, *, max_passes: int = 64) -> np.ndarray:
    """Làm dày qua nền: C=Aᶜ, mảnh C, trả về Cᶜ (C4 § Thickening).

    Slide khuyến nghị làm dày bằng mảnh nền rồi phần bù; có thể sinh điểm rời — kết quả
    giữ các pixel 8-liên thông với tập sau mở rộng.

    Parameters
    ----------
    binary
        Tập A.
    max_passes
        Giới hạn vòng lặp cho bước mảnh nền.

    Returns
    -------
    np.ndarray
        ``bool`` tập dày hơn A.

    Examples
    --------
    >>> line = np.zeros((5, 5), dtype=np.uint8); line[2, 1:4] = 255
    >>> thickening(line)[2, 2]
    np.True_
    """
    mask = require_binary_image(binary)
    # Làm dày A ↔ làm mảnh phần bù Aᶜ (đối xứng morphology slide)
    comp = ~mask
    thinned_bg = thinning(comp, max_passes=max_passes)
    thick = ~thinned_bg
    # Loại pixel cô lập không 8-liên thông với phần còn lại (hậu xử lý slide)
    return _drop_isolated_points(thick, connectivity=8)


def _drop_isolated_points(mask: np.ndarray, connectivity: Connectivity) -> np.ndarray:
    """Xoá pixel foreground không có láng giềng foreground (theo connectivity)."""
    # Chọn lưới láng giềng theo định nghĩa liên thông 4 hoặc 8 (segmentation / morphology)
    if connectivity == 4:
        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    h, w = mask.shape
    out = mask.copy()
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            count = 0
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                    count += 1
            # Điểm đơn lẻ không láng giềng foreground — artefact sau thickening
            if count == 0:
                out[y, x] = False
    return out


def skeleton(
    binary: np.ndarray,
    se: np.ndarray,
    *,
    return_subsets: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
    """Xương S(A) theo Lantuejoul: Sₖ = Aₖ − (Aₖ ∘ B), S = ⋃ Sₖ (C4 § Skeleton).

    Với Aₖ là A co k lần liên tiếp bởi B, K là bước cuối trước khi A rỗng.

    Parameters
    ----------
    binary
        Tập A.
    se
        Phần tử cấu trúc B (lẻ).
    return_subsets
        Nếu True, trả thêm danh sách các Sₖ.

    Returns
    -------
    skel
        ``bool`` tổng hợp xương.
    subsets
        Khi ``return_subsets``, danh sách ``bool`` S₀…S_{K−1}.

    Examples
    --------
    >>> line = np.zeros((5, 9), dtype=np.uint8); line[2, 1:8] = 255
    >>> sk = skeleton(line, np.ones((3, 3), dtype=np.uint8))
    >>> sk.shape
    (5, 9)
    """
    mask = require_binary_image(binary)
    elem = _check_se(se)
    a_k = mask.copy()
    skel = np.zeros_like(mask)
    subsets: list[np.ndarray] = []
    # Co dần A cho tới rỗng, cộng dồn lõi sau mở tại mỗi mức
    while np.any(a_k):
        opened = _binary_open(a_k, elem)
        # Sₖ = phần Aₖ bị “mất” khi mở — gai nhỏ và lõi 1-pixel rơi vào đây (skeleton morphology)
        sk = a_k & ~opened
        skel |= sk
        subsets.append(sk.copy())
        # Bước kế: Aₖ₊₁ = Aₖ eroded — chuỗi co đa tỷ lệ Lantuejoul
        a_k = _binary_erode(a_k, elem)
    if return_subsets:
        return skel, subsets
    return skel
