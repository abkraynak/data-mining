# naive_bayes.py

def nb_model(path: str, values: list, gender: list):
    with open(path, 'r') as csvfile:
        dr = csv.reader(csvfile)
        for row in dr:
            if row[0] != 'ResponseId': # Skip the first row
                if row[47] != 'NA' and row[7] != 'NA':
                    values[age_dict[row[7]]].append(int(row[47]))
                if (row[39] == 'Man' or row[39] == 'Woman' or row[39] == 'Non-binary, genderqueer, or gender non-conforming') and row[7] != 'NA':
                    gender[age_dict[row[7]]][gender_dict[row[39]]] += 1

def stats_calc(l: list) -> list:
    # Returns list containing mean, stdev, and variance from list
    res = []
    res.append(st.mean(l))
    res.append(st.stdev(l))
    res.append(st.variance(l))
    return res

def get_nb_stats(l) -> list:
    res = []
    for element in l:
        res.append(stats_calc(element))
    return res

def calc_nb(x: float, mean: float, stdev: float, var: float) -> float:
    # Naive Bayes probabilistic density formula
    return (1 / (mth.sqrt(2 * mth.pi) * stdev) ) * mth.exp(-((x - mean) ** 2) / (2 * var))

def nb(x: float, stats) -> list:
    res = []
    for l in stats:
        res.append(calc_nb(x, l[0], l[1], l[2]))
    return res

def nominal_prob(query: str, data: list, lookup: dict) -> float:
    # Calculates probability for nominal attributes under Naive Bayes rules
    return data[lookup[query]] / sum(data)

def nominal_prob_list(query: str, data: list, lookup: dict) -> list:
    res = []
    for i in range(len(data)):
        res.append(nominal_prob(query, data[i], lookup))
    return res

def column_sum(lst):  
    return [sum(i) for i in zip(*lst)]

def totalclasslabelprob(data: list, lookup: dict, query: str):
    #print(sum(list(data[j] for j in range (len(data[0])))))
    return (sum(data[lookup[query]])) / sum(column_sum(data))

def get_final_probs(l1: list, l2: list):
    res = []

    for i in range(len(l1)):
        res.append(l1[i] * l2[i])

    return res

def get_category(lookup: dict, pos: int) -> str:
    itemsList = lookup.items()
    for item in itemsList:
        if item[1] == pos:
            return item[0]

def validate(sal: int, gen: str, target: str):
    print(salary_age1stcode)
    salary_probs = nb(sal, get_nb_stats(salary_age1stcode))
    gender_probs = nominal_prob_list(gen, gender_age1stcode, gender_dict)
    res = get_final_probs(salary_probs, gender_probs)
    return get_category(age_dict, res.index(max(res))) == target

def get_accuracy(path: str):
    total = 0
    valid = 0
    with open(path, 'r') as csvfile:
        dr = csv.reader(csvfile)
        print('open')
        for row in dr:
            if row[0] != 'ResponseId': # Skip the first row
                if total > 50: 
                    break
                if validate(int(row[47]), row[39], row[7]):
                    valid += 1
                total += 1
        print(valid)
    return valid // total

def NBone_att(dr, values: list):
    print(dr)
    for row in dr:
        print(row)
        values[age(row[7])].append(row[47])
    print(values)