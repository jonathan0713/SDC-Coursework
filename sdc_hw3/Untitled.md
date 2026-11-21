2.2 (b)
```python
import random
from enum import Enum

class Weather(str, Enum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"


# 格式為 {當前天氣: {下一個天氣: 機率}}
transitions = {
    Weather.SUNNY:  {Weather.SUNNY: 0.8, Weather.CLOUDY: 0.2, Weather.RAINY: 0.0},
    Weather.CLOUDY: {Weather.SUNNY: 0.4, Weather.CLOUDY: 0.4, Weather.RAINY: 0.2},
    Weather.RAINY:  {Weather.SUNNY: 0.2, Weather.CLOUDY: 0.6, Weather.RAINY: 0.2}
}

def weather_sim(init_weather, days):
    """
    隨機生成天氣序列的模擬器
    :param init_weather: 初始天氣狀態
    :param days: 模擬的天數
    :return: 天氣狀態列表
    """
    current_weather = init_weather
    sequence = [current_weather]
    
    for _ in range(days - 1):
        # 取得當前天氣對應的下一天天氣選項與權重
        options = list(transitions[current_weather].keys())
        weights = list(transitions[current_weather].values())
        
        # 根據權重進行隨機抽樣 [cite: 85, 87]
        current_weather = random.choices(options, weights=weights, k=1)[0]
        sequence.append(current_weather)
        
    return sequence

# 執行測試：從晴天開始模擬 10 天
if __name__ == "__main__":
    result = weather_sim(Weather.SUNNY, 10)
    print("產生的天氣序列：")
    for i, weather in enumerate(result, start=1):
        print(f"第 {i} 天: {weather.value}")
```
![[Screenshot from 2026-03-25 20-03-04.png]]

2.2 (c)
```python
import random
from enum import Enum, auto  
from collections import Counter

class Weather(str, Enum):   
    SUNNY = "sunny"          
    CLOUDY = "cloudy"
    RAINY = "rainy"


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

```

![[Screenshot from 2026-03-25 20-03-27.png]]