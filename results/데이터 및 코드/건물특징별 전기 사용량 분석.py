import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
# Load the data
file_path = 'result.csv'
data = pd.read_csv(file_path)

# Define the features to analyze with appropriate bins
features = {
    '연면적(㎡)': '연면적',
    '건폐율(%)': '건폐율',
    '용적률(%)': '용적률',
    '지상층수': '지상층수',
    '경과개월수': '경과 개월 수',
    '세대수': '세대 수'
}

# '연면적(㎡)' 로그 변환
data['연면적(㎡)'] = np.log1p(data['연면적(㎡)'])

def calculate_bin_size(min_value, max_value, num_bins=25):
    return np.ceil((max_value - min_value) / num_bins)

# 각 특성과 전기 사용량 간의 관계 시각화
for feature, feature_name in features.items():
    min_value = data[feature].min()
    max_value = data[feature].max()
    bin_size = calculate_bin_size(min_value, max_value)
    bins = np.arange(min_value, max_value + bin_size, bin_size)
    data_binned = data.copy()
    data_binned[f'{feature}_binned'] = pd.cut(data[feature], bins=bins)
    bin_means = data_binned.groupby(f'{feature}_binned')['전기사용량'].mean()

    plt.figure(figsize=(10, 6))
    plt.bar(bin_means.index.astype(str), bin_means.values, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{feature_name}에 따른 전기 사용량', fontsize=14)
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('전기 사용량', fontsize=12)
    plt.grid(True)
    plt.tight_layout(pad=2.0)
    plt.show()