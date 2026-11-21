import random
from enum import Enum, auto  # 將 StrEnum 改為 Enum
from collections import Counter

class Weather(str, Enum):    # 透過繼承 str 來模擬 StrEnum
    SUNNY = "sunny"          # 在 3.10 中建議直接賦予字串值
    CLOUDY = "cloudy"
    RAINY = "rainy"

# 根據作業提供的轉移矩陣定義
# P(tomorrow | today)
transitions = {
    Weather.SUNNY: {Weather.SUNNY: 0.8, Weather.CLOUDY: 0.2, Weather.RAINY: 0.0},
    Weather.CLOUDY: {Weather.SUNNY: 0.4, Weather.CLOUDY: 0.4, Weather.RAINY: 0.2},
    Weather.RAINY: {Weather.SUNNY: 0.2, Weather.CLOUDY: 0.6, Weather.RAINY: 0.2}
}

def weather_sim(init_weather, days):
    current_weather = init_weather
    sequence = [current_weather]
    
    for _ in range(days - 1):
        # 取得當前天氣可能的下一狀態與權重
        options = list(transitions[current_weather].keys())
        weights = list(transitions[current_weather].values())
        
        # 隨機抽樣
        next_weather = random.choices(options, weights=weights, k=1)[0]
        sequence.append(next_weather)
        current_weather = next_weather
        
    return sequence

# --- 執行模擬 ---
total_days = 1_000_000
result_seq = weather_sim(Weather.SUNNY, total_days)

# --- 計算平穩分佈 (c) ---
counts = Counter(result_seq)
stationary_dist = {state: count / total_days for state, count in counts.items()}

print(f"模擬 {total_days} 天後的平穩分佈：")
for state in Weather:
    print(f"{state.value.capitalize()}: {stationary_dist[state]:.4f}")

# 驗證理論值 (9/14, 4/14, 1/14)
print(f"\n理論值驗證：")
print(f"Sunny (9/14): {9/14:.4f}, Cloudy (4/14): {4/14:.4f}, Rainy (1/14): {1/14:.4f}")
