import random
from enum import Enum

# 在 Python 3.10 中，我們使用繼承 (str, Enum) 來達到類似 StrEnum 的效果
class Weather(str, Enum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"

# 根據作業提供的狀態轉移矩陣 
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