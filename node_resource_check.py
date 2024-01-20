import random
import pandas as pd

# 호스트의 CPU, 메모리, GPU 사용량을 측정하는 함수
def measure_host_resources():
    # CPU 사용량 측정 (랜덤 값)
    cpu_usage = random.uniform(0, 100)
    
    # 메모리 사용량 측정 (랜덤 값)
    memory_usage = random.uniform(0, 100)
    
    # GPU 사용량 측정 (랜덤 값)
    gpu_usage = random.uniform(0, 100)
    
    return cpu_usage, memory_usage, gpu_usage

# 호스트 리소스 측정 및 누적 수치 DataFrame 생성
def generate_accumulated_resource_dataframe(num_measurements):
    # 초기값 설정
    cpu_accumulated = 0
    memory_accumulated = 0
    gpu_accumulated = 0
    
    # 측정 결과를 저장할 리스트
    cpu_measurements = []
    memory_measurements = []
    gpu_measurements = []
    
    for i in range(num_measurements):
        # 호스트 리소스 측정
        cpu_usage, memory_usage, gpu_usage = measure_host_resources()
        
        # 누적 수치 업데이트
        cpu_accumulated += cpu_usage
        memory_accumulated += memory_usage
        gpu_accumulated += gpu_usage
        
        # 측정 결과 저장
        cpu_measurements.append(cpu_accumulated)
        memory_measurements.append(memory_accumulated)
        gpu_measurements.append(gpu_accumulated)
    
    # DataFrame 생성
    df = pd.DataFrame({
        'CPU(Usage)': cpu_measurements,
        'MEMORY(Usage)': memory_measurements,
        'GPU(Usage)': gpu_measurements
    })
    
    return df

# 호스트 리소스 측정 및 누적 수치 DataFrame 생성 예시
num_measurements = 10  # 측정할 횟수
df = generate_accumulated_resource_dataframe(num_measurements)
print(df)
