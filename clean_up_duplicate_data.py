#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, sys
from pathlib import Path
from shutil import move

ORIG_RE = re.compile(r'^\s*Ảnh gốc:\s*(.*?)\s*\(')
DUP_RE  = re.compile(r'^\s*Ảnh trùng:\s*(.*?)\s*\(')
PAIR_RE = re.compile(r'^\s*Cặp trùng lặp\s*\d+:')

def parse_pairs(list_file: Path):
    """Trả về danh sách (orig_path, dup_path) theo từng cặp trong file."""
    pairs = []
    orig = None
    with list_file.open('r', encoding='utf-8') as f:
        for line in f:
            if PAIR_RE.match(line):
                orig = None  # bắt đầu cặp mới
                continue
            m = ORIG_RE.match(line)
            if m:
                orig = m.group(1).strip()
                continue
            m = DUP_RE.match(line)
            if m and orig:
                dup = m.group(1).strip()
                pairs.append((orig, dup))
                orig = None  # kết thúc cặp
    return pairs

def ensure_under_root(root: Path, p: str) -> Path:
    """Ghép đường dẫn trong file (tương đối) vào root; nếu p đã là tuyệt đối thì dùng luôn."""
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)

def main():
    ap = argparse.ArgumentParser(description="Delete duplicate images, keep originals.")
    ap.add_argument("--list", required=True, type=Path, help="Đường dẫn file danh sách trùng (txt).")
    ap.add_argument("--root", required=True, type=Path, help="Thư mục gốc chứa dataset.")
    ap.add_argument("--doit", action="store_true", help="Thực thi xóa/di chuyển (mặc định chỉ in).")
    ap.add_argument("--move-to", type=Path, default=None,
                    help="Nếu đặt, di chuyển ảnh trùng vào thư mục này thay vì xóa.")
    ap.add_argument("--write-log", type=Path, default=Path("deleted_duplicates.txt"),
                    help="Ghi danh sách đã xóa/di chuyển vào file này.")
    args = ap.parse_args()

    if not args.list.exists():
        print(f"[ERR] Không thấy file danh sách: {args.list}", file=sys.stderr)
        sys.exit(1)
    if not args.root.exists():
        print(f"[ERR] Không thấy thư mục root: {args.root}", file=sys.stderr)
        sys.exit(1)

    pairs = parse_pairs(args.list)
    if not pairs:
        print("[WARN] Không tìm thấy cặp trùng nào trong file.")
        sys.exit(0)

    to_delete = []
    for orig_rel, dup_rel in pairs:
        orig = ensure_under_root(args.root, orig_rel)
        dup  = ensure_under_root(args.root, dup_rel)

        # Bảo vệ: nếu đường dẫn trùng nhau thì bỏ qua
        if orig.resolve().as_posix().lower() == dup.resolve().as_posix().lower() if dup.exists() and orig.exists() else False:
            print(f"[SKIP] orig == dup: {dup}")
            continue

        to_delete.append((orig, dup))

    # Dùng set để không xóa cùng một file nhiều lần nếu xuất hiện ở nhiều cặp
    uniq_dups = []
    seen = set()
    for _, dup in to_delete:
        key = dup.resolve() if dup.exists() else dup
        if key not in seen:
            seen.add(key)
            uniq_dups.append(dup)

    print(f"[INFO] Tổng cặp đọc được: {len(pairs)}")
    print(f"[INFO] Ảnh trùng duy nhất cần xử lý: {len(uniq_dups)}")

    # In danh sách
    for dup in uniq_dups:
        action = "MOVE" if args.move_to else "DELETE"
        print(f"{action} -> {dup}")

    # Thực thi
    if args.doit:
        log_lines = []
        if args.move_to:
            args.move_to.mkdir(parents=True, exist_ok=True)

        for dup in uniq_dups:
            if not dup.exists():
                print(f"[MISS] Không thấy file: {dup}")
                continue
            try:
                if args.move_to:
                    # Giữ cấu trúc thư mục tương đối trong thùng rác
                    rel = dup.relative_to(args.root) if dup.is_relative_to(args.root) else Path(dup.name)
                    dest = (args.move_to / rel)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    move(str(dup), str(dest))
                    log_lines.append(f"MOVED\t{dup}\t->\t{dest}")
                else:
                    dup.unlink()
                    log_lines.append(f"DELETED\t{dup}")
            except Exception as e:
                print(f"[ERR] {dup}: {e}")

        # Ghi log
        try:
            with args.write_log.open('w', encoding='utf-8') as fw:
                fw.write("\n".join(log_lines))
            print(f"[DONE] Ghi log: {args.write_log}")
        except Exception as e:
            print(f"[ERR] Ghi log: {e}")
    else:
        print("[DRY-RUN] Thêm --doit để thực thi. Có thể dùng --move-to ./_trash để di chuyển thay vì xóa.")

if __name__ == "__main__":
    main()
