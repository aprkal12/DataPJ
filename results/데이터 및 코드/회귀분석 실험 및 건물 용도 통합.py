def usage():
    #건물 용도 통합 및  용도별 사용량 시각화
    import pandas as pd
    import matplotlib.pyplot as plt
    import re
    plt.rcParams['font.family'] = 'Malgun Gothic'
    # 데이터 로드
    data = pd.read_csv('result.csv')

    # 용도를 소문자로 변환하고 공백을 제거하는 등 전처리 작업
    data['기타용도_전처리'] = data['기타용도'].str.lower().str.replace(' ', '')

    # 용도 통합 규칙 정의
    def classify_usage(usage):
        if '주거' in usage:
            return '주거'
        elif '상업' in usage:
            return '상업'
        elif '사무' in usage:
            return '사무'
        elif '공업' in usage:
            return '공업'
        elif '교육' in usage:
            return '교육'
        elif '의료' in usage:
            return '의료'
        elif '숙박' in usage:
            return '숙박'
        else:
            return '기타'

    # 용도를 통합하는 새로운 컬럼 추가
    data['통합용도'] = data['기타용도_전처리'].apply(classify_usage)

    # '통합용도'별 전기 사용량의 평균 계산
    usage_electricity_usage = data.groupby('통합용도')['전기사용량'].mean().reset_index()

    # 결과 출력
    print(usage_electricity_usage)

    # '통합용도'별 전기 사용량의 평균 시각화
    plt.figure(figsize=(12, 8))
    plt.barh(usage_electricity_usage['통합용도'], usage_electricity_usage['전기사용량'], color='skyblue')
    plt.xlabel('평균 전기사용량')
    plt.ylabel('통합용도')
    plt.title('통합 용도별 평균 전기사용량')
    plt.gca().invert_yaxis()
    plt.show()




# import pandas as pd
# import matplotlib.pyplot as plt
# 
# # 데이터 로드
# data = pd.read_csv('result.csv')

# # 주 구조별 전기 사용량의 평균 계산
# structure_electricity_usage = data.groupby('주구조')['전기사용량'].mean().reset_index()

# # 결과 출력
# print(structure_electricity_usage)

# # 주 구조별 전기 사용량의 평균 시각화
# plt.figure(figsize=(12, 8))
# plt.barh(structure_electricity_usage['주구조'], structure_electricity_usage['전기사용량'], color='skyblue')
# plt.xlabel('평균 전기사용량')
# plt.ylabel('주구조')
# plt.title('주 구조별 평균 전기사용량')
# plt.gca().invert_yaxis()
# plt.show()


# # K-means clustering
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 데이터 로드
# data = pd.read_csv('result.csv')

# # 분석에 사용할 컬럼 선택
# features = ['대지면적(㎡)', '건축면적(㎡)', '연면적(㎡)', '건폐율(%)', '용적률(%)', 
#             '용적률산정연면적(㎡)', '높이', '지상층수', '지하층수', '세대수', 
#             '가구수', '호수', '경과년수', '경과개월수']

# # 결측값이 있는 행 제거
# cleaned_data_multivariate = data[features].dropna()

# # 특성 정규화
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(cleaned_data_multivariate)

# # K-means 클러스터링 적용
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(X_scaled)

# # 클러스터 라벨을 정리된 데이터에 추가
# cleaned_data_multivariate['Cluster'] = clusters

# # 클러스터 결과를 원본 데이터와 병합
# data_with_clusters = data.merge(cleaned_data_multivariate[['Cluster']], left_index=True, right_index=True, how='left')

# # PCA를 사용하여 클러스터 시각화
# pca = PCA(n_components=2)
# pca_components = pca.fit_transform(X_scaled)

# plt.figure(figsize=(10, 8))
# sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=clusters, palette='viridis')
# plt.title('PCA of Clusters')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.show()




# 로그변환 정규화
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from tqdm import tqdm

# # Load your data
# data_cleaned = pd.read_csv('result.csv')

# # Select relevant columns for the analysis
# features = ['대지면적(㎡)', '건축면적(㎡)', '연면적(㎡)', '건폐율(%)', '용적률(%)', 
#             '용적률산정연면적(㎡)', '높이', '지상층수', '지하층수', '세대수', 
#             '가구수', '호수', '경과년수', '경과개월수']
# target = '전기사용량'

# # Remove rows with missing values in the relevant columns
# cleaned_data_multivariate = data_cleaned[features + [target]].dropna()

# # Apply log transformation
# X_log = np.log1p(cleaned_data_multivariate[features])
# y_log = np.log1p(cleaned_data_multivariate[target])

# # Polynomial Regression
# poly_features = PolynomialFeatures(degree=2)
# X_poly = poly_features.fit_transform(X_log)
# X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y_log, test_size=0.2, random_state=42)

# print("Training Polynomial Regression model...")
# poly_model = LinearRegression()
# poly_model.fit(X_train_poly, y_train_poly)
# y_pred_poly = poly_model.predict(X_test_poly)

# mse_poly = mean_squared_error(np.expm1(y_test_poly), np.expm1(y_pred_poly))
# r2_poly = r2_score(np.expm1(y_test_poly), np.expm1(y_pred_poly))

# # Decision Tree Regression
# X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

# print("Training Decision Tree model...")
# tree_model = DecisionTreeRegressor(random_state=42)
# tree_model.fit(X_train_log, y_train_log)
# y_pred_tree = tree_model.predict(X_test_log)

# mse_tree = mean_squared_error(np.expm1(y_test_log), np.expm1(y_pred_tree))
# r2_tree = r2_score(np.expm1(y_test_log), np.expm1(y_pred_tree))

# # Random Forest Regression
# forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# print("Training Random Forest model...")
# for i in tqdm(range(1, 101)):
#     forest_model.set_params(n_estimators=i)
#     forest_model.fit(X_train_log, y_train_log)

# y_pred_forest = forest_model.predict(X_test_log)

# mse_forest = mean_squared_error(np.expm1(y_test_log), np.expm1(y_pred_forest))
# r2_forest = r2_score(np.expm1(y_test_log), np.expm1(y_pred_forest))

# # Print the results
# print(f"Polynomial Regression MSE: {mse_poly}, R2: {r2_poly}")
# print(f"Decision Tree Regression MSE: {mse_tree}, R2: {r2_tree}")
# print(f"Random Forest Regression MSE: {mse_forest}, R2: {r2_forest}")




# 정규화 적용, 다항회귀, 의사결정트리, 랜덤포레스트 회귀 모델을 사용하여 전기사용량 예측
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from tqdm import tqdm

# # Load your data
# data_cleaned = pd.read_csv('result.csv')

# # Select relevant columns for the analysis
# features = ['대지면적(㎡)', '건축면적(㎡)', '연면적(㎡)', '건폐율(%)', '용적률(%)', 
#             '용적률산정연면적(㎡)', '높이', '지상층수', '지하층수', '세대수', 
#             '가구수', '호수', '경과년수', '경과개월수']
# target = '전기사용량'

# # Remove rows with missing values in the relevant columns
# cleaned_data_multivariate = data_cleaned[features + [target]].dropna()

# # Normalize the features
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(cleaned_data_multivariate[features])
# y = cleaned_data_multivariate[target]

# # Polynomial Regression
# poly_features = PolynomialFeatures(degree=2)
# X_poly = poly_features.fit_transform(X_scaled)
# X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# print("Training Polynomial Regression model...")
# poly_model = LinearRegression()
# poly_model.fit(X_train_poly, y_train_poly)
# y_pred_poly = poly_model.predict(X_test_poly)

# mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
# r2_poly = r2_score(y_test_poly, y_pred_poly)

# # Decision Tree Regression
# X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# print("Training Decision Tree model...")
# tree_model = DecisionTreeRegressor(random_state=42)
# tree_model.fit(X_train_scaled, y_train_scaled)
# y_pred_tree = tree_model.predict(X_test_scaled)

# mse_tree = mean_squared_error(y_test_scaled, y_pred_tree)
# r2_tree = r2_score(y_test_scaled, y_pred_tree)

# # Random Forest Regression
# forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# print("Training Random Forest model...")
# for i in tqdm(range(1, 101)):
#     forest_model.set_params(n_estimators=i)
#     forest_model.fit(X_train_scaled, y_train_scaled)

# y_pred_forest = forest_model.predict(X_test_scaled)

# mse_forest = mean_squared_error(y_test_scaled, y_pred_forest)
# r2_forest = r2_score(y_test_scaled, y_pred_forest)

# # Print the results
# print(f"Polynomial Regression MSE: {mse_poly}, R2: {r2_poly}")
# print(f"Decision Tree Regression MSE: {mse_tree}, R2: {r2_tree}")
# print(f"Random Forest Regression MSE: {mse_forest}, R2: {r2_forest}")
