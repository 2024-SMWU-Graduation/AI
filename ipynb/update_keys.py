import torch
from collections import OrderedDict

def update_state_dict_keys(weights_path, new_weights_path, prefix="model."):
    """
    가중치의 키 값에 prefix를 추가하여 새로 저장하는 함수.
    Args:
        weights_path (str): 기존 가중치 파일 경로
        new_weights_path (str): 변경된 가중치를 저장할 파일 경로
        prefix (str): 각 키 앞에 추가할 접두사 (기본값: "model.")
    """
    # 기존 가중치 로드
    state_dict = torch.load(weights_path, map_location="cpu")

    # 새로운 state_dict 생성
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = prefix + key  # 접두사 추가
        new_state_dict[new_key] = value

    # 새로운 가중치 저장
    torch.save(new_state_dict, new_weights_path)
    print(f"Updated state_dict saved to {new_weights_path}")


weights_path = "/Users/mks/Documents/GitHub/AI/model/ResNet18_final_best.pth"
new_weights_path = "/Users/mks/Documents/GitHub/AI/model/ResNet18_final_best_updated.pth"

# 키 값 변환
update_state_dict_keys(weights_path, new_weights_path)
