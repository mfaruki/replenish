from replenish.get_data import bbc_scrape

df_indian = bbc_scrape.category_bbc_data('Indian')

def time_cleaner(time):
    convert_to_list = time.replace(" and ","").split(' hr')
    if len(convert_to_list)==2:
        hours = convert_to_list[0]
        if convert_to_list[1] == "":
            mins = 0
        else:
            mins = convert_to_list[1]
    else:
        hours = 0
        mins = convert_to_list[0]

    total_mins = (int(hours)*60) + int(mins)
    return total_mins

def serving_cleaner(serving):
    convert_to_list = serving.replace("-"," to ").split()
    extract_digits = [int(x) for x in convert_to_list if x.isdigit()]
    smallest_digit = min(extract_digits)
    return smallest_digit

df_indian = df_indian[df_indian['servings'] != 'None']
df_indian['prep_times'] = df_indian['prep_times'].apply(time_cleaner)
df_indian['cooking_times'] = df_indian['cooking_times'].apply(time_cleaner)
df_indian['servings'] = df_indian['servings'].apply(serving_cleaner)

df_indian
