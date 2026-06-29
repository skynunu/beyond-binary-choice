#!/usr/bin/env python3
"""
claude-4.5-sonnet 에이전트 결과 병합 스크립트

1) FAILED_claude45sonnet_..._idx0_51_valid52.csv (df1, 152행)
   - claude4sonnet_* (p1~p5 vote, judgment 6개) 열 삭제
   - 같은 파일에 덮어쓰기 저장 (사전 .bak 백업)
2) df2 = claude45sonnet_..._idx52_151_valid99.csv (100행)
   - df1의 비어있는 rows(인덱스 52~151)의 claude45sonnet 6개 열 값을
     행 일치 검증 후 df2에서 채워 넣는다.
3) 추가로 유효 행 수(valid{N})를 반영한 리네이밍 사본도 생성.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

FAILED_CSV = BASE_DIR / "FAILED_claude45sonnet_last_dataset_embed_filtered_th_0.85_agent_152_idx0_51_valid52.csv"
RESUMED_CSV = BASE_DIR / "claude45sonnet_last_dataset_embed_filtered_th_0.85_agent_152_idx52_151_valid99.csv"

COLS_TO_DROP = [
    "claude4sonnet_p1_vote",
    "claude4sonnet_p2_vote",
    "claude4sonnet_p3_vote",
    "claude4sonnet_p4_vote",
    "claude4sonnet_p5_vote",
    "claude4sonnet_judgment",
]

COLS_TO_FILL = [
    "claude45sonnet_p1_vote",
    "claude45sonnet_p2_vote",
    "claude45sonnet_p3_vote",
    "claude45sonnet_p4_vote",
    "claude45sonnet_p5_vote",
    "claude45sonnet_judgment",
]

# 행 일치(key) 검증에 쓸 열 (원본 데이터 식별용). 존재하는 열만 사용.
KEY_CANDIDATE_COLS = [
    "agent_dilemma",
    "agent_ab_dilemma",
    "option_a",
    "option_b",
]


def norm(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def is_empty_cell(v) -> bool:
    if pd.isna(v):
        return True
    return str(v).strip() == ""


def main() -> None:
    if not FAILED_CSV.exists():
        print(f"파일 없음: {FAILED_CSV}", file=sys.stderr)
        sys.exit(1)
    if not RESUMED_CSV.exists():
        print(f"파일 없음: {RESUMED_CSV}", file=sys.stderr)
        sys.exit(1)

    print(f"=== 1단계: {FAILED_CSV.name} 읽기 & 백업 ===")
    bak_path = FAILED_CSV.with_suffix(FAILED_CSV.suffix + ".bak")
    shutil.copy2(FAILED_CSV, bak_path)
    print(f"  백업: {bak_path.name}")

    df1 = pd.read_csv(FAILED_CSV, encoding="utf-8-sig")
    df2 = pd.read_csv(RESUMED_CSV, encoding="utf-8-sig")
    print(f"  df1 (FAILED): shape={df1.shape}")
    print(f"  df2 (idx52_151): shape={df2.shape}")

    # ── 2단계: claude4sonnet 열 삭제 후 덮어쓰기 ────────────────────
    print("\n=== 2단계: claude4sonnet 6개 열 삭제 ===")
    existing_drop = [c for c in COLS_TO_DROP if c in df1.columns]
    missing_drop = [c for c in COLS_TO_DROP if c not in df1.columns]
    if missing_drop:
        print(f"  [WARN] df1에 없는 열: {missing_drop}")
    df1 = df1.drop(columns=existing_drop)
    print(f"  삭제: {existing_drop}")
    print(f"  삭제 후 df1.shape: {df1.shape}")

    df1.to_csv(FAILED_CSV, index=False, encoding="utf-8-sig")
    print(f"  덮어쓰기 저장: {FAILED_CSV.name}")

    # ── 3단계: claude45sonnet 6개 열 존재 확인 ──────────────────────
    print("\n=== 3단계: 병합 대상 열 확인 ===")
    for c in COLS_TO_FILL:
        if c not in df1.columns:
            print(f"  [오류] df1에 '{c}' 열 없음 → 중단", file=sys.stderr)
            sys.exit(1)
        if c not in df2.columns:
            print(f"  [오류] df2에 '{c}' 열 없음 → 중단", file=sys.stderr)
            sys.exit(1)
    print("  OK")

    # ── 4단계: 행 매칭 전략 결정 ────────────────────────────────────
    # 가장 직관적 매칭: df1 행 52..(52+len(df2)-1)  ↔  df2 행 0..len(df2)-1
    # 파일명(idx52_151)과 일치. 내용 기반 검증을 추가로 수행한다.
    print("\n=== 4단계: 행 매칭 검증 ===")
    target_start = 52
    target_end = target_start + len(df2)  # exclusive
    if target_end > len(df1):
        print(
            f"  [오류] df1 길이({len(df1)})가 {target_end}보다 작음 → 중단",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  매칭 범위: df1[{target_start}:{target_end}]  ↔  df2[0:{len(df2)}]")

    key_cols = [c for c in KEY_CANDIDATE_COLS if c in df1.columns and c in df2.columns]
    print(f"  검증에 사용할 key 열: {key_cols}")

    mismatches = []
    for j in range(len(df2)):
        i = target_start + j
        for kc in key_cols:
            if norm(df1.iloc[i][kc]) != norm(df2.iloc[j][kc]):
                mismatches.append((i, j, kc))
                break

    if mismatches:
        print(f"  [오류] 행 불일치 {len(mismatches)}건 발견 → 첫 3건:")
        for i, j, kc in mismatches[:3]:
            v1 = norm(df1.iloc[i][kc])[:80]
            v2 = norm(df2.iloc[j][kc])[:80]
            print(f"    df1[{i}].{kc}={v1!r}")
            print(f"    df2[{j}].{kc}={v2!r}")
        sys.exit(1)

    print(f"  모든 {len(df2)}행 일치 확인")

    # ── 5단계: 비어있는 셀만 df2 값으로 채우기 ─────────────────────
    print("\n=== 5단계: 비어있는 셀만 채우기 ===")
    fill_stats = {c: 0 for c in COLS_TO_FILL}
    overwrite_conflicts = []

    for j in range(len(df2)):
        i = target_start + j
        for c in COLS_TO_FILL:
            cur = df1.iloc[i][c]
            new = df2.iloc[j][c]
            if is_empty_cell(cur):
                df1.at[i, c] = new
                if not is_empty_cell(new):
                    fill_stats[c] += 1
            else:
                # 원래 df1에 이미 값이 있는데 df2와 다르면 기록
                if not is_empty_cell(new) and norm(cur) != norm(new):
                    overwrite_conflicts.append((i, c, norm(cur)[:40], norm(new)[:40]))

    for c in COLS_TO_FILL:
        filled = fill_stats[c]
        total_nonempty = int(
            (df1[c].notna() & (df1[c].astype(str).str.strip() != "")).sum()
        )
        print(f"  {c}: +{filled} 채움, 전체 non-empty={total_nonempty}/{len(df1)}")

    if overwrite_conflicts:
        print(
            f"\n  [주의] df1에 이미 값이 있으면서 df2와 다른 셀 {len(overwrite_conflicts)}건 "
            f"(덮어쓰지 않고 유지). 첫 3건:"
        )
        for i, c, v1, v2 in overwrite_conflicts[:3]:
            print(f"    row={i}, col={c}, df1={v1!r}, df2={v2!r}")

    # ── 6단계: 최종 저장 ─────────────────────────────────────────
    print("\n=== 6단계: 최종 저장 ===")
    df1.to_csv(FAILED_CSV, index=False, encoding="utf-8-sig")
    print(f"  덮어쓰기: {FAILED_CSV.name}")

    # 유효 행 수에 맞춘 리네이밍 사본 저장
    judgment_s = df1["claude45sonnet_judgment"].astype(str).str.strip()
    valid_rows = int(
        ((judgment_s != "") & (judgment_s.str.lower() != "nan")).sum()
    )
    renamed_name = (
        f"claude45sonnet_last_dataset_embed_filtered_th_0.85_agent_152_"
        f"idx0_151_valid{valid_rows}.csv"
    )
    renamed_path = BASE_DIR / renamed_name
    df1.to_csv(renamed_path, index=False, encoding="utf-8-sig")
    print(f"  리네이밍 사본: {renamed_name} (valid={valid_rows}/{len(df1)})")

    print("\n완료.")


if __name__ == "__main__":
    main()
