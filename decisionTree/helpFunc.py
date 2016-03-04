import operator

G_Accuracy = 1e-8

def createDataSet(file_name):
    fobj = open(file_name,'r')

    line = fobj.readline()
    feature_attr = (line.strip()).split('\t')

    data_set = []
    for line in fobj:
        data_set.append(line.strip().split('\t'))

    return (feature_attr,data_set)

def splitDataSet(data_set,axis,value):
    sub_data_set = []

    for line in data_set:
        if line[axis] == value:
            sub_line = line[:axis]
            sub_line.extend(line[axis+1:])
            sub_data_set.append(sub_line)

    return sub_data_set

def majorityValue(data_set):
    value_dict = {}

    for line in data_set:
        if line[-1] not in value_dict.keys():
            value_dict[line[-1]] = 0
        value_dict[line[-1]] += 1

    new_dict = sorted(value_dict.items(),key = operator.itemgetter(1),reverse = True)

    return new_dict[0][0]

