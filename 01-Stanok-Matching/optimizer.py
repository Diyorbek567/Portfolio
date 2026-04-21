import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations

# Excel faylini yuklash
excel_file = 'SAG-CarpetAllocation202505230900_.xlsx'
xls = pd.ExcelFile(excel_file)

programs_df = pd.read_excel(xls, sheet_name='Programs')
dashboard_df = pd.read_excel(xls, sheet_name='Dashboard-1')

programs_df = programs_df.copy()
dashboard_df = dashboard_df.copy()

# Fayl nomidan sana olish
filename_date_str = excel_file.split('SAG-CarpetAllocation')[-1].split('.')[0][:8]
file_date = datetime.strptime(filename_date_str, "%Y%m%d")
remaining_hours = datetime(file_date.year, file_date.month, file_date.day, 9, 0) + timedelta(days=1)

# Optimal kombinatsiyani topish funksiyasi
def find_best_combination(programs, time_limit, max_comb_len=6):
    best_combo = []
    best_total_time = 0

    for r in range(1, max_comb_len + 1):
        for combo in combinations(programs, r):
            total_time = sum([p[1] for p in combo])
            if total_time >= time_limit and (best_total_time == 0 or total_time < best_total_time):
                best_combo = combo
                best_total_time = total_time
    return best_combo, best_total_time

# Natija uchun lug‘at
result = {}

# Har bir stanok uchun hisoblash
for stanok in [f"T{str(i).zfill(2)}" for i in range(1, 36)]:
    row = dashboard_df[dashboard_df['Станок'] == stanok]
    if row.empty:
        continue

    end_time_str = row['Очередь окончание ткания'].values[0]
    if pd.isna(end_time_str):
        continue

    end_time = pd.to_datetime(end_time_str, dayfirst=True)
    total_hours = (remaining_hours - end_time).total_seconds() / 3600
    if total_hours <= 0:
        continue

    program_rows = programs_df[programs_df['Станок'] == stanok].copy()
    program_rows['Полотно, см'] = program_rows['Полотно, см'].fillna(0)

    selected_programs = []
    used_time = 0
    unique_values = sorted(program_rows['Полотно, см'].unique())

    for val in unique_values:
        related_programs = program_rows[program_rows['Полотно, см'] == val]
        program_list = list(zip(related_programs['Program'], related_programs['Время, ч']))
        val_total_time = sum([p[1] for p in program_list])
        remaining_time = total_hours - used_time

        if val_total_time >= remaining_time:
            best_combo, combo_time = find_best_combination(program_list, remaining_time)
            selected_programs.extend([p[0] for p in best_combo])
            used_time += combo_time
            break
        else:
            selected_programs.extend([p[0] for p in program_list])
            used_time += val_total_time
            if used_time >= total_hours:
                break

    ordered_selected_programs = sorted(selected_programs, key=lambda prog: program_rows[program_rows['Program'] == prog]['Время, ч'].values[0])

    result[stanok] = {
        'Bo‘sh vaqt (soat)': round(total_hours, 2),
        'Ajratilgan vaqt (soat)': round(used_time, 2),
        'Tanlangan programlar soni': len(selected_programs),
        'Tanlangan programlar': ordered_selected_programs
    }

# Natijani chiqarish
for stanok, info in result.items():
    print(f"{stanok}:")
    for key, value in info.items():
        if key == 'Tanlangan programlar':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    print("-" * 40)
print("\n========== POLОTNO YIG‘INDISI ==========\n")

def polotno_kvm(row):
    try:
        path = str(row['Path definition'])
        widths = path.split('x')
        num_stoyka = len(widths)

        polotno = row['Полотно, см']
        dlina=row['Длина, см']
        dlina1 = row['Длина см']
        
        if num_stoyka == 1:
            return (polotno * dlina) / 10000
        
        elif num_stoyka == 2:
            dlina2 = row['Длина см.1']
            base = (polotno * dlina) / 10000
            extra1=(dlina-dlina1)/ 10000
            extra2=(dlina-dlina2)/ 10000
            return base + extra1+extra2

        elif num_stoyka == 3:
            dlina2 = row['Длина см.1']
            dlina3 = row['Длина см.2']
            base = (polotno * dlina) / 10000
            extra1=(dlina-dlina1)/ 10000
            extra2=(dlina-dlina2)/ 10000
            extra3=(dlina-dlina3)/ 10000
        
            return base + extra1 + extra2+extra3

        else:
            return np.nan
    except:
        return np.nan

# Sana olish
filename_date_str = excel_file.split('SAG-CarpetAllocation')[-1].split('.')[0][:8]
formatted_date = datetime.strptime(filename_date_str, "%Y%m%d").strftime("%d.%m.%Y")

# Yangi ustun nomlari
percent_col = f"{formatted_date} (%)"
prog_kvm_col = f"{formatted_date} (KVM)"
polotno_kvm_col = f"{formatted_date} (Polotno)"

# Yakuniy natijalarni tayyorlash
final_result = {
    'Станок': [],
    percent_col: [],
    prog_kvm_col: [],
    polotno_kvm_col: []
}

# Har bir stanok uchun hisoblash
for stanok in [f"T{str(i).zfill(2)}" for i in range(1, 36)]:
    selected_programs = result.get(stanok, {}).get('Tanlangan programlar', [])
    if not selected_programs:
        continue

    program_rows = programs_df[(programs_df['Станок'] == stanok) & 
                               (programs_df['Program'].isin(selected_programs))]

    total_prog_kvm = 0
    total_polotno_kvm = 0

    for _, row in program_rows.iterrows():
        shirina = row.get('Ширина станка')
        dlina = row.get('Длина, см')
        polotno = row.get('Полотно, см')

        if pd.isna(shirina) or pd.isna(dlina) or pd.isna(polotno):
            continue

        prog_kvm = (shirina * dlina) / 10000
         # polotno_kvm funktsiyasini chaqirish
        polotno_kvm_value = polotno_kvm(row)

        total_prog_kvm += prog_kvm
        total_polotno_kvm += polotno_kvm_value

    if total_prog_kvm == 0:
        continue

    foiz = round((total_polotno_kvm / total_prog_kvm) * 100, 2)

    final_result['Станок'].append(stanok)
    final_result[percent_col].append(foiz)
    final_result[prog_kvm_col].append(round(total_prog_kvm, 2))
    final_result[polotno_kvm_col].append(round(total_polotno_kvm, 2))

# DataFramega o‘tkazish va Excelga yozish
df_final = pd.DataFrame(final_result)
df_final.to_excel("result0_.xlsx", index=False)

print(f"✅ Natijalar quyidagi ustunlar bilan 'result.xlsx' faylga saqlandi:\n→ {percent_col}, {prog_kvm_col}, {polotno_kvm_col}")
