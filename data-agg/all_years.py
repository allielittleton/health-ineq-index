import pandas as pd
import statistics
import numpy as np
import math

SAMPLE_SIZE = 100000.0


def analyze(df):
    age = list(df["Age (x)"])
    deaths = df["Number of deaths d(x,n)"]
    p_AaD = list(deaths / SAMPLE_SIZE)

    mode = max(p_AaD)

    p_AaD_list = list(p_AaD)
    index = p_AaD_list.index(mode)  # find index of mode value

    AaD_mode = age[index]  # find age at that index

    # option A
    if AaD_mode == 100:
        p_AaD_list[index] = 0
        new_mode = max(p_AaD_list)
        new_index = p_AaD_list.index(new_mode)
        AaD_mode = age[new_index]

    # option B
    # if AaD_mode == 100:
    # start = age.index(90)
    # end = age.index(100)
    # y = array(p_AaD_list[start:end])
    # x = array(range(len(y)))
    # params, covs = curve_fit(func1, x, y)
    # a, b, c = params[0], params[1], params[2]

    # start_new = end
    # end_new = end + 11
    # next_y = func1(start_new, a, b, c)
    # p_AaD_list[start_new] = next_y
    # for next_x in range(start_new+1, end_new):
    #  next_y = func1(next_x, a, b, c)
    #  p_AaD_list.append(next_y)
    # new_mode = max(p_AaD_list)
    #  new_index = p_AaD_list.index(new_mode)
    # AaD_mode = age[new_index]
    # print(AaD_mode)

    abs_dev = absolute_deviation(age, p_AaD, AaD_mode)
    standard_dev = standard_deviation(age, p_AaD, AaD_mode)

    return abs_dev, standard_dev


# Our idea
def absolute_deviation(AaD_list, p_AaD_list, AaD_m):
    sum = 0.0
    n = len(AaD_list)
    for i in range(n):
        AaD = AaD_list[i]
        p_AaD = p_AaD_list[i]  # probability of death at current age
        sum = sum + (abs(AaD - AaD_m)*p_AaD)
    return sum


# Standard deviation
def standard_deviation(AaD_list, p_AaD_list, AaD_m):
    sum = 0.0
    n = len(AaD_list)
    for i in range(n):
        AaD = AaD_list[i]
        distance_sq = (AaD - AaD_m) * (AaD - AaD_m)  # (x_i - x_m)^2
        sum = sum + (distance_sq * p_AaD_list[i])
    return math.sqrt(sum)


# mode = 2%, all other vals = (98/99)%
def perfect_inequality_sim(mode, calc_method):
    sum = 0.0
    for AaD in range(101):
        p_AaD = float(98/99)
        if AaD == mode:
            p_AaD = .02
        if calc_method == 'absolute':
            sum = sum + (abs(AaD - mode) * p_AaD)
        else:
            distance_sq = (AaD - mode) * (AaD - mode)  # (x_i - x_m)^2
            sum = sum + math.sqrt(distance_sq * p_AaD)
    return sum


def main():
    print("here")
    dystopian_val_absdev = perfect_inequality_sim(0, 'absolute')
    dystopian_val_stdev = perfect_inequality_sim(0, 'standard')
    input_df = pd.read_excel('Country_Data_2010-2020.xlsx')

    output_df = pd.DataFrame()

    countries = []
    country_codes = list(input_df["ISO3 Alpha-code"].unique())
    stdev_vals = []
    abs_dev_vals = []
    years = list(input_df["Year"].unique())
    df_years = []
    for year in years:
        print("in loop")
        year_df = input_df[input_df["Year"] == year]

        for country_code in country_codes:
            country_df = year_df[year_df["ISO3 Alpha-code"] == country_code]
            print(country_df)
            abs_dev, standard_dev = analyze(country_df)
            stdev_vals.append((standard_dev/dystopian_val_stdev)*10**3)
            abs_dev_vals.append((abs_dev/dystopian_val_absdev)*10**3)

            df_years.append(list(country_df["Year"])[0])
            countries.append(list(country_df["Country"])[0])

    output_df["Year"] = df_years
    output_df["Country"] = countries
    output_df["Absolute Deviation Index Value"] = abs_dev_vals
    output_df["Standard Deviation Index Value"] = stdev_vals

    output_df.to_excel("Inequality Index Values by Year and Country.xlsx")

    output_df1 = output_df.sort_values(by=['Absolute Deviation Index Value'])
    output_df1.to_excel("Inequality Index Values by Country 2010-2020 (sorted by abs. dev).xlsx")

    output_df2 = output_df.sort_values(by=['Standard Deviation Index Value'])
    output_df2.to_excel("Inequality Index Values by Country 2010-2020 (sorted by stand. dev).xlsx")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
