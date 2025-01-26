# 1. 전기 데이터 월별 통합
import pandas as pd
def data_preproc3():
    import pandas as pd

    # CSV 파일 불러오기
    # df = pd.read_csv('서울특별시전기에너지2023_processed.csv')

    # 사용년월 열을 문자열로 변환 후, datetime 형식으로 변환
    # df['사용_년월'] = df['사용_년월'].astype(str).str.strip('.0')  # 소수점과 그 뒤의 0을 제거
    # df['사용_년월'] = pd.to_datetime(df['사용_년월'], format='%Y%m', errors='coerce')

    # # 전기 사용량 데이터를 월별로 합산
    # monthly_usage = df.groupby(df['사용_년월'].dt.to_period('M')).sum().reset_index()

    # # 사용년월 열을 다시 datetime 형식으로 변환
    # monthly_usage['사용_년월'] = monthly_usage['사용_년월'].dt.to_timestamp()

    # # 변환된 데이터 확인
    # print(monthly_usage[['사용_년월', '사용량(KWh)']].head(50))

    # 결과를 새로운 CSV 파일로 저장
    # monthly_usage.to_csv('new_elec.csv', index=False)

    monthly_usage = pd.read_csv('result.csv')
    print(monthly_usage[['사용년월', '전기사용량']].head())
    print("월별로 합산된 전기 사용량 데이터가 'monthly_usage.csv'로 저장되었습니다.")

# 2. 통합을 위한 대지_위치 컬럼 생성
def create_juso():
    import pandas as pd

    # # CSV 파일 불러오기
    # df = pd.read_csv('D:\\dataeng\\buildings.csv')

    # # 시군구, 법정동, 번, 지 열을 합쳐서 주소 형식으로 만들기
    # df['대지_위치'] = df['시도'] + ' ' + df['시군구'] + ' ' + df['법정동'] + ' ' + df['번'].astype(str) + '-' + df['지'].astype(str) + '번지'

    # 새로운 CSV 파일로 저장
    # df.to_csv('D:\\dataeng\\new_buildings.csv', index=False)
    df = pd.read_csv('result.csv')
    print(df['대지위치'].head())
    print("주소가 추가된 CSV 파일이 'output_file.csv'로 저장되었습니다.")

# 3. 강남구 데이터만 추출
def only_gangnam():
    

    # # Load the CSV file
    # file_path = 'new_buildings.csv'
    # df = pd.read_csv(file_path)

    # # Filter rows where 시군구 is '강남구'
    # filtered_df = df[df['시군구'] == '강남구']

    # Display the filtered dataframe
    filtered_df = pd.read_csv('result.csv')
    print(filtered_df.head())

    # Optionally, save the filtered dataframe to a new CSV file
    # filtered_df.to_csv('only_gangnamgu.csv', index=False)

# 4. 데이터 통합
def data_integrated():
    # # CSV 파일 불러오기
    # df1 = pd.read_csv('only_gangnamgu.csv')
    # df2 = pd.read_csv('monthly_elec.csv')

    # # 두 데이터프레임을 '대지위치'를 기준으로 병합 (left join)
    # merged_df = pd.merge(df1, df2[['대지_위치','사용_년월', '사용량(KWh)']], on='대지_위치', how='left')

    # # 결과를 새로운 CSV 파일로 저장
    # # merged_df.to_csv('real_last_monthly_elec_gangnam_fusion.csv', index=False)

    # print("병합된 데이터가 'merged_output.csv'로 저장되었습니다.")

    # df = pd.read_csv('real_last_monthly_elec_gangnam_fusion.csv')
    merged_df = pd.read_csv('result.csv')
    print(merged_df[['대지위치', '사용년월', '전기사용량']].head())

# 전기 사용량 데이터 null 삭제
def check_null():
    # file_path = 'D:\\dataeng\\result.csv'
    file_path = 'D:\\dataeng\\real_last_monthly_elec_gangnam_fusion.csv'
    # # # file_path2 = 'D:\\dataeng\\서울특별시전기에너지2023_processed.csv'

    data = pd.read_csv(file_path, encoding='utf-8-sig')
    # # data2 = pd.read_csv(file_path2, encoding='utf-8-sig')

    # print(data[['대지_위치', '사용_년월', '사용량(KWh)']].head(50))
    # print("=====================================")
    # print(data2.columns)
    # print(data['시도'].head())
    # print(data2['대지_위치'].head())
    # print(data.loc[:, '사용_년월'])
    # print(data.shape)

    # 각 열의 null 값 개수 확인
    null_counts = data.isnull().sum()
    # 데이터 프레임의 전체 행 개수
    total_counts = len(data)

    # 결과를 데이터 프레임으로 결합
    result = pd.DataFrame({
        'null_count': null_counts,
        'total_count': [total_counts] * len(data.columns)
    })
    print(result)
    # 전기사용량 컬럼에서 NaN 값을 포함하는 행 삭제
    data = data.dropna(subset=['사용량(KWh)'])

    # 각 열의 null 값 개수 확인
    null_counts = data.isnull().sum()
    # 데이터 프레임의 전체 행 개수
    total_counts = len(data)

    # 결과를 데이터 프레임으로 결합
    result = pd.DataFrame({
        'null_count': null_counts,
        'total_count': [total_counts] * len(data.columns)
    })
    print(result)

# 건물 노후도 계산을 위한 컬럼 추가
def add_coulumn():
    import pandas as pd
    from datetime import datetime

    # CSV 파일 불러오기
    # df = pd.read_csv('real_real_last_data.csv')
    # df = pd.read_csv('test1.csv')
    # print(df[['경과년수', '경과개월수']].head())

    # # 사용승인일 열에서 .0 제거
    # df['사용승인일'] = df['사용승인일'].astype(str).str.replace('.0', '')

    # # 사용승인일을 datetime 형식으로 변환
    # df['사용승인일'] = pd.to_datetime(df['사용승인일'], format='%Y%m%d', errors='coerce')

    # # 현재 날짜
    # current_date = datetime.now()

    # # 몇 년이 지났는지 계산
    # df['경과년수'] = df['사용승인일'].apply(lambda x: (current_date - x).days / 365.25 if pd.notnull(x) else None)

    # # 몇 개월이 지났는지 계산
    # df['경과개월수'] = df['사용승인일'].apply(lambda x: (current_date - x).days / 30.4375 if pd.notnull(x) else None)

    # # 경과개월수를 소수점 첫째자리까지 반올림
    # df['경과개월수'] = df['경과개월수'].round(1)
    # df['경과년수'] = df['경과년수'].round(1)

    # 결과를 새로운 CSV 파일로 저장
    df = pd.read_csv('result.csv')
    # df.to_csv('test1.csv', index=False)
    print(df[['경과년수', '경과개월수']].head())

    print("경과년수와 경과개월수가 추가된 데이터가 'output_with_years_and_months.csv'로 저장되었습니다.")


# 5. 건물 특징별 전기 사용량 평균 계산
def test1():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams['font.family'] = 'Malgun Gothic'
    # Load the data
    file_path = 'result.csv'
    data = pd.read_csv(file_path)

    # 분석 대상 특성 정의
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

# 6. 건물 용도별 전기 사용량 평균 계산
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

# 계절별 전기 사용량
def seasonal_usage():
    
    import pandas as pd
    import plotly.express as px
    import matplotlib.font_manager as fm

    data_cleaned = pd.read_csv('result.csv', parse_dates=['사용년월'])

    # 사용년월 열에서 연도와 월을 추출
    data_cleaned['year'] = data_cleaned['사용년월'].dt.year
    data_cleaned['month'] = data_cleaned['사용년월'].dt.month

    # 계졀별로 분류
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'

    data_cleaned['season'] = data_cleaned['month'].apply(get_season)

    # 계절별 전기 사용량의 평균 계산
    seasonal_usage = data_cleaned.groupby('season')['전기사용량'].mean().reset_index()

    # 계절 순서대로 정렬렬
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    seasonal_usage['season'] = pd.Categorical(seasonal_usage['season'], categories=season_order, ordered=True)
    seasonal_usage = seasonal_usage.sort_values('season')

    # 계절별 전기 사용량의 평균 시각화
    fig = px.bar(seasonal_usage, x='season', y='전기사용량', title='계절별 전기사용량 평균')

    # 시각화 옵션 설정
    fig.update_layout(
        xaxis_title='계절',
        yaxis_title='전기사용량',
        title_font=dict(size=24),
        font=dict(family='Malgun Gothic', size=18)
    )

    fig.show()

if __name__ == '__main__':
    import plotly.express as px
    import numpy as np
    data = pd.read_csv('D:\\dataeng\\result.csv')
    # basic_stats = df.describe(include='all')
    data['전기사용량_log'] = np.log1p(data['전기사용량'])  # np.log1p는 log(1+x)를 계산하여 0인 값에 대해 안정적으로 로그를 계산할 수 있게 함

# 로그 변환된 전기사용량에 대한 기초 통계 분석
    electricity_usage_log_stats = data['전기사용량_log'].describe()
    print(electricity_usage_log_stats)
    # print(df.head())

    # df2 = pd.read_csv('D:\\dataeng\\서울특별시전기에너지2023.csv', encoding='cp949')
    # print(df2.head())

    # data_preproc3()
    # create_juso()
    # only_gangnam()
    # data_integrated()
    # check_null()
    # add_coulumn()
    # test1()
    # usage()
    # seasonal_usage()
