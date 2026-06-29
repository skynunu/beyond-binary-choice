import json
import csv
import os
from typing import List, Dict, Any


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """주어진 경로의 .jsonl 파일을 읽어 리스트[dict]로 반환."""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """리스트[dict]를 CSV로 저장. 모든 행의 key 합집합을 헤더로 사용."""
    if not rows:
        # 빈 리스트인 경우라도 헤더 없는 빈 파일을 만들어 둔다.
        open(path, "w", encoding="utf-8").close()
        return

    # 모든 key의 합집합을 헤더로 사용 (순서는 안정적으로 정렬)
    fieldnames = sorted({k for row in rows for k in row.keys()})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge_jsonl_files(file1: str, file2: str) -> str:
    """
    두 개의 .jsonl 파일을 (file1 먼저, 그 뒤 file2) 순서로 병합하여
    같은 디렉터리에 CSV로 저장하고, 그 CSV 경로를 반환한다.
    """
    # 데이터 읽기
    data1 = read_jsonl(file1)
    data2 = read_jsonl(file2)
    merged = data1 + data2  # file1이 첫째, 이어서 file2

    # 출력 파일명 구성: 두 파일 이름을 이어 붙이고 확장자를 .csv 로
    dirpath = os.path.dirname(file1)
    base1 = os.path.basename(file1)
    base2 = os.path.basename(file2)

    # 공통 접두어/패턴이 이미 있으니, 단순히 '_' 로 이어 붙이되 .jsonl 제거
    name1_no_ext = os.path.splitext(base1)[0]
    name2_no_ext = os.path.splitext(base2)[0]
    out_name = f"{name1_no_ext}__{name2_no_ext}.csv"
    out_path = os.path.join(dirpath, out_name)

    write_csv(out_path, merged)
    return out_path


if __name__ == "__main__":
    # 1) advisor 병합
    advisor_f1 = "/home/jchan/Desktop/research_codes/conflict_resolutions/5_basic_prompting_llm_response/non_compromise_outputs/advisor_gpt5_non_compromise_s0-e29_30.jsonl"
    advisor_f2 = "/home/jchan/Desktop/research_codes/conflict_resolutions/5_basic_prompting_llm_response/non_compromise_outputs/advisor_gpt5_non_compromise_s30-e200_170.jsonl"
    advisor_csv = merge_jsonl_files(advisor_f1, advisor_f2)
    print(f"advisor 파일 병합 완료: {advisor_csv}")

    # 2) agent 병합
    agent_f1 = "/home/jchan/Desktop/research_codes/conflict_resolutions/5_basic_prompting_llm_response/non_compromise_outputs/agent_gpt5_non_compromise_s0-e91_92.jsonl"
    agent_f2 = "/home/jchan/Desktop/research_codes/conflict_resolutions/5_basic_prompting_llm_response/non_compromise_outputs/agent_gpt5_non_compromise_s92-e200_108.jsonl"
    agent_csv = merge_jsonl_files(agent_f1, agent_f2)
    print(f"agent 파일 병합 완료: {agent_csv}")

