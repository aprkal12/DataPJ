# builgings csv 파일 오픈
import pandas as pd


def read_data():
    #파일 경로
    file_path = 'D:\\dataeng\\국토교통부_건물에너지_전기에너지+(2023년)\\MART_KEY_01\\mart_key_01_2023.txt'

    # 파일 읽기
    data = pd.read_csv(file_path, sep="\t", engine='python', encoding="cp949", header=None)

    # 데이터가 한 컬럼으로 읽혀졌는지 확인
    # print(data.head())

    # 데이터가 한 컬럼으로 읽혀졌다면, 첫 번째 컬럼을 분리
    if data.shape[1] == 1:
        print('한 컬럼으로 읽혀진 데이터를 분리합니다.')
        data_split = data[0].str.split('|', expand=True)

        # 나누어진 데이터의 첫 몇 행을 확인
        print(data_split.head())

        # 새로운 컬럼명을 지정 (실제 컬럼 개수에 맞게 조정)
        new_columns = ['순번', '사용_년월', '대지_위치', '도로명_대지_위치', '시군구_코드', '법정동_코드', '대지_구분_코드', '번', '지', '새주소_일련번호', '새주소_도로코드', '새주소_지상지하_코드', '새주소_본_번', '새주소_부_번', '사용량(KWh)']

        # 데이터 프레임에 새로운 컬럼명 지정
        data_split.columns = new_columns

        # 새로운 데이터 프레임 출력
        print(data_split.head())

        # 데이터프레임을 CSV 파일로 저장
        output_file_path = 'D:\\dataeng\\국토교통부_건물에너지_전기에너지+(2023년)\\MART_KEY_01\\mart_key_01_2023_processed.csv'
        data_split.to_csv(output_file_path, index=False, encoding="utf-8-sig")

        print(f"파일이 성공적으로 저장되었습니다: {output_file_path}")
    else:
        print("데이터가 예상과 다르게 읽혀졌습니다. 데이터를 확인하세요.")

# ==========================================================

def check_null():
    file_path = 'D:\\dataeng\\new_buildings.csv'
    file_path = 'D:\\dataeng\\buildings_elec_fusion2.csv'
    # # # file_path2 = 'D:\\dataeng\\서울특별시전기에너지2023_processed.csv'

    data = pd.read_csv(file_path, encoding='utf-8-sig')
    # # data2 = pd.read_csv(file_path2, encoding='utf-8-sig')

    print(data[['대지_위치', '사용_년월', '사용량(KWh)']].head(50))
    # print("=====================================")
    # print(data2.columns)
    print(data['시도'].head())
    # print(data2['대지_위치'].head())
    # print(data.loc[:, '사용_년월'])
    print(data.shape)

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

    print(data['건물연식'].head(50))

def create_juso():
    import pandas as pd

    # CSV 파일 불러오기
    df = pd.read_csv('D:\\dataeng\\buildings.csv')

    # 시군구, 법정동, 번, 지 열을 합쳐서 주소 형식으로 만들기
    df['대지_위치'] = df['시도'] + ' ' + df['시군구'] + ' ' + df['법정동'] + ' ' + df['번'].astype(str) + '-' + df['지'].astype(str) + '번지'

    # 새로운 CSV 파일로 저장
    df.to_csv('D:\\dataeng\\new_buildings.csv', index=False)

    print("주소가 추가된 CSV 파일이 'output_file.csv'로 저장되었습니다.")

def data_join():
    import pandas as pd

    # CSV 파일 불러오기
    df1 = pd.read_csv('D:\\dataeng\\new_buildings.csv')
    df2 = pd.read_csv('D:\\dataeng\\서울특별시전기에너지2023_processed.csv')

    # 두 데이터프레임을 '대지위치'를 기준으로 병합 (left join)
    merged_df = pd.merge(df1, df2[['대지_위치', '사용_년월', '사용량(KWh)']], on='대지_위치', how='left')

    # 결과를 새로운 CSV 파일로 저장
    merged_df.to_csv('buildings_elec_fusion2.csv', index=False)

    print("병합된 데이터가 'merged_output.csv'로 저장되었습니다.")

def data_preproc():
    import pandas as pd
    from datetime import datetime

    # CSV 파일 불러오기
    df = pd.read_csv('buildings_elec_fusion.csv')

    # 예외 처리를 사용하여 사용승인일을 datetime 형식으로 변환
    def convert_to_datetime(date_str):
        try:
            return pd.to_datetime(date_str, format='%Y%m%d')
        except (ValueError, TypeError):
            return pd.NaT

    # 사용승인일 열의 공백 제거
    df['사용승인일'] = df['사용승인일'].astype(str).str.strip()

    # 사용승인일 열을 datetime 형식으로 변환
    df['사용승인일'] = df['사용승인일'].apply(convert_to_datetime)

    # 현재 날짜
    current_date = datetime.now()

    # 몇 년이 지났는지 계산
    df['건물연식'] = df['사용승인일'].apply(lambda x: (current_date - x).days / 365.25 if pd.notnull(x) else None)

    # 경과년수를 자연수로 변환
    df['건물연식'] = df['건물연식'].fillna(0).astype(int)

    # 결과를 새로운 CSV 파일로 저장
    df.to_csv('new_buildings_elec_fusion.csv', index=False)

    print("경과년수가 자연수로 변환된 데이터가 'output_with_years_int.csv'로 저장되었습니다.")

def data_preproc2():
    import pandas as pd

    # CSV 파일 불러오기
    df = pd.read_csv('new_buildings_elec_fusion.csv')

    # 주구조의 고유 값 찾기 (NaN 제외)
    unique_structures = df['주구조'].dropna().unique()
    print("고유한 주구조:", unique_structures)

    # NaN 값을 제외한 후 주구조를 숫자로 변환
    df['주구조_코드'] = df['주구조'].astype('category').cat.codes

    # NaN 값을 갖는 행은 주구조_코드에서 -1로 표시될 수 있습니다.
    # 이를 NaN으로 변경
    df.loc[df['주구조'].isna(), '주구조_코드'] = None

    # 변환된 데이터 확인
    print(df[['주구조', '주구조_코드']].head())

    # 결과를 새로운 CSV 파일로 저장
    df.to_csv('all_new_buildings_elec_fusion.csv', index=False)

    print("주구조 코드가 추가된 데이터가 'output_with_main_structure_codes.csv'로 저장되었습니다.")

def corr():
    # 상관관계 출력
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 깨짐 설정
    # CSV 파일 불러오기
    df = pd.read_csv('all_new_buildings_elec_fusion.csv')

    # structure_cols = 
    # 필요한 열만 선택
    df_selected = df[['주구조_코드','대지면적(㎡)', '건축면적(㎡)', '연면적(㎡)','건폐율(%)', '용적률(%)','높이','지상층수', '지하층수','총동연면적','세대수','가구수','호수','건물연식', '사용량(KWh)']]

    # 데이터의 결측값 제거
    df_selected = df_selected.dropna()

    # 상관 행렬 계산
    correlation_matrix = df_selected.corr()

    # 소수점 셋째자리까지 표시하도록 포맷 설정
    formatted_corr_matrix = correlation_matrix.applymap(lambda x: f'{x:.3f}')

    # 상관 행렬 출력
    print(formatted_corr_matrix)

    # 상관 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

def data_preproc3():
    import pandas as pd

    # CSV 파일 불러오기
    df = pd.read_csv('서울특별시전기에너지2023_processed.csv')

    # 사용년월 열을 문자열로 변환 후, datetime 형식으로 변환
    df['사용_년월'] = df['사용_년월'].astype(str).str.strip('.0')  # 소수점과 그 뒤의 0을 제거
    df['사용_년월'] = pd.to_datetime(df['사용_년월'], format='%Y%m', errors='coerce')

    # 전기 사용량 데이터를 월별로 합산
    monthly_usage = df.groupby(df['사용_년월'].dt.to_period('M')).sum().reset_index()

    # 사용년월 열을 다시 datetime 형식으로 변환
    monthly_usage['사용_년월'] = monthly_usage['사용_년월'].dt.to_timestamp()

    # 변환된 데이터 확인
    print(monthly_usage[['사용_년월', '사용량(KWh)']].head(50))

    # 결과를 새로운 CSV 파일로 저장
    monthly_usage.to_csv('new_elec.csv', index=False)

    print("월별로 합산된 전기 사용량 데이터가 'monthly_usage.csv'로 저장되었습니다.")

# import pandas as pd

# df = pd.read_csv('real_real_last_data.csv')
# print(df[['사용년월', '대지위치', '전기사용량', '가스사용량']].head(50))
# # print(df[['대지위치', '사용_년월', '전기사용량', '가스사용량']].head(50))

if __name__ == '__main__':
    read_data()
    # check_null()
    # create_juso()
    # data_join()
    # data_preproc()
    # data_preproc2()
    # corr()
    # data_preproc3()

