#-*- coding: utf-8 -*-
import csv
import numpy as np
import pickle

def read_data(filename):
    fp = open(filename, 'r')
    csv_reader = csv.reader(fp)

    # csv_reader에서 2차원 list로 변환
    # for문을 사용하여 하나씩 새로운 list에 추가
    samsung_data = []
    for row in csv_reader:
        samsung_data.append(row)
    
    # 사용하지 않을 데이터 제거
    print(samsung_data[0])
    del samsung_data[0]

    
    dataset = []
    for data in samsung_data:
        '''
        학습에는 날짜를 사용하지 않을 것이므로 날짜를 빼고
        null로 이루어진 missing data도 제거
        문자열로 이루어진 숫자들을 float형으로 변경하여
        새로운 2차원 numpy matrix를 만듭니다.
        '''
        data_point = []
        is_null = False
        for element in data[1:]:
            if element == 'null':
                is_null = True
            else:
                data_point.append(float(element))

        if is_null:
            continue

        dataset.append(np.array(data_point))
    dataset = np.array(dataset)

    print(dataset)
    print(np.shape(dataset))
    return dataset


def time_slicer(dataset):
    '''
    우리가 풀어야 할 문제는 30일 동안의 주식데이터를 이용하여
    미래 30일의 주가 움직임을 예측하는 것이므로
    데이터를 이 문제에 맞도록 가공해야 합니다.
    hint:
        input 차원: n * 30 * 6
        output 차원: n * 30 * 6
    '''
    start = 0
    end = 30
    inputs = []
    labels = []
    # windowing
    while end + 30 <= len(dataset):
        input_segment = dataset[start:end]
        label_segment = dataset[end:end+30]

        inputs.append(input_segment)
        labels.append(label_segment)

        start += 1
        end += 1

    inputs = np.array(inputs)
    labels = np.array(labels)

    print(np.shape(inputs))
    print(np.shape(labels))

    # 총 4107개의 segment 중에 4000개를 trainset, 나머지를 testset으로 사용할 것입니다.
    # 만들어진 데이터셋을 나눠봅시다.
    train_input = inputs[:4000]
    train_label = labels[:4000]

    test_input = inputs[4000:]
    test_label = labels[4000:]

    return train_input, train_label, test_input, test_label


if __name__ == '__main__':
    dataset = read_data('005930.KS.csv')
    train_input, train_label, test_input, test_label = time_slicer(dataset)
    print(np.shape(train_input))
    print(np.shape(train_label))
    print(np.shape(test_input))
    print(np.shape(test_label))
