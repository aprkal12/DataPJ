import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 깨짐 설정


def struc():
    import pandas as pd
    import matplotlib.pyplot as plt

    # CSV 파일 불러오기
    df = pd.read_csv('gangnam_electricity_usage.csv')

    # 전기사용량 컬럼에서 NaN 값을 포함하는 행 삭제
    df = df.dropna(subset=['전기사용량'])

    # 전기사용량을 kWh에서 MWh로 변환
    df['전기사용량'] = df['전기사용량'] / 1000  # kWh -> MWh

    # 주구조별 전기사용량 합산
    structure_usage = df.groupby('주구조')['전기사용량'].sum().reset_index()

    # 주구조 코드를 문자열로 변환하여 시각화의 명확성 증가
    structure_usage['주구조'] = structure_usage['주구조'].astype(str)

    # 주구조별 전기사용량을 내림차순으로 정렬
    structure_usage = structure_usage.sort_values(by='전기사용량', ascending=False)

    # 주구조별 전기사용량 시각화
    plt.figure(figsize=(14, 8))
    bars = plt.bar(structure_usage['주구조'], structure_usage['전기사용량'], color='skyblue')
    plt.xlabel('주구조 코드')
    plt.ylabel('전기사용량 (MWh)')
    plt.title('주구조별 전기사용량 합계')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 막대 그래프 위에 수치 표시
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', ha='center', va='bottom')

    # 그래프 보여주기
    plt.show()


def data_integrated():
    # CSV 파일 불러오기
    df1 = pd.read_csv('only_gangnamgu.csv')
    df2 = pd.read_csv('monthly_elec.csv')

    # 두 데이터프레임을 '대지위치'를 기준으로 병합 (left join)
    merged_df = pd.merge(df1, df2[['대지_위치','사용_년월', '사용량(KWh)']], on='대지_위치', how='left')

    # 결과를 새로운 CSV 파일로 저장
    merged_df.to_csv('real_last_monthly_elec_gangnam_fusion.csv', index=False)

    print("병합된 데이터가 'merged_output.csv'로 저장되었습니다.")

    df = pd.read_csv('real_last_monthly_elec_gangnam_fusion.csv')
    print(df[['대지_위치', '사용_년월', '사용량(KWh)']].head(50))


    import pandas as pd

    # Load the CSV file
    file_path = 'new_buildings.csv'
    df = pd.read_csv(file_path)

    # Filter rows where 시군구 is '강남구'
    filtered_df = df[df['시군구'] == '강남구']

    # Display the filtered dataframe
    print(filtered_df)

    # Optionally, save the filtered dataframe to a new CSV file
    filtered_df.to_csv('only_gangnamgu.csv', index=False)

def coulumn_fix():
    import pandas as pd

    # CSV 파일 불러오기
    df = pd.read_csv('gas2023.csv')

    # 사용년월 열을 문자열로 변환 후, datetime 형식으로 변환
    df['사용년월'] = df['사용년월'].astype(str).str.strip('.0')  # 소수점과 그 뒤의 0을 제거
    df['사용년월'] = pd.to_datetime(df['사용년월'], format='%Y%m', errors='coerce')

    # 전기 사용량 데이터를 대지위치별, 월별로 합산
    monthly_usage = df.groupby(['대지위치', df['사용년월'].dt.to_period('M')]).agg({'사용량(KWh)': 'sum'}).reset_index()

    # 사용년월 열을 다시 datetime 형식으로 변환
    monthly_usage['사용년월'] = monthly_usage['사용년월'].dt.to_timestamp()

    # 결과 데이터프레임 확인
    print(monthly_usage.head(50))

    # 결과를 새로운 CSV 파일로 저장
    monthly_usage.to_csv('monthly_gas.csv', index=False)

    print("대지위치별 월별로 합산된 전기 사용량 데이터가 'monthly_usage_by_location.csv'로 저장되었습니다.")

def add_coulumn():
    import pandas as pd
    from datetime import datetime

    # CSV 파일 불러오기
    df = pd.read_csv('real_real_last_data.csv')
    df = pd.read_csv('test1.csv')
    print(df[['경과년수', '경과개월수']].head())

    # 사용승인일 열에서 .0 제거
    df['사용승인일'] = df['사용승인일'].astype(str).str.replace('.0', '')

    # 사용승인일을 datetime 형식으로 변환
    df['사용승인일'] = pd.to_datetime(df['사용승인일'], format='%Y%m%d', errors='coerce')

    # 현재 날짜
    current_date = datetime.now()

    # 몇 년이 지났는지 계산
    df['경과년수'] = df['사용승인일'].apply(lambda x: (current_date - x).days / 365.25 if pd.notnull(x) else None)

    # 몇 개월이 지났는지 계산
    df['경과개월수'] = df['사용승인일'].apply(lambda x: (current_date - x).days / 30.4375 if pd.notnull(x) else None)

    # 경과개월수를 소수점 첫째자리까지 반올림
    df['경과개월수'] = df['경과개월수'].round(1)
    df['경과년수'] = df['경과년수'].round(1)

    # 결과를 새로운 CSV 파일로 저장
    df.to_csv('test1.csv', index=False)

    print("경과년수와 경과개월수가 추가된 데이터가 'output_with_years_and_months.csv'로 저장되었습니다.")

if __name__ == '__main__':
    add_coulumn()
    # coulumn_fix()
    # data_integrated()
    # struc()
