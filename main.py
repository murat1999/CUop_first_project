import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

file = sys.argv[1]


def calculate(arr, arr2, arr3):
    for element in range(len(arr)):
        if element == 0:
            arr[element] = 0
        if element == len(arr) - 1:
            arr[element] = float(arr2[element] / arr3[element])
        else:
            arr[element] = float(arr2[element + 1] - arr2[element - 1]) / float(arr3[element + 1] - arr3[element - 1])
    return arr


def find_local_maxima_and_minima(arr):
    mx = []
    mn = []
    arr_length = len(arr)
    if arr[0] > arr[1]:
        mx.append(0)
    elif arr[0] < arr[1]:
        mn.append(0)
    for ind in range(1, arr_length - 1):
        if arr[ind - 1] < arr[ind] > arr[ind + 1]:
            mx.append(ind)
        elif arr[ind - 1] > arr[ind] < arr[ind + 1]:
            mn.append(ind)
    if arr[-1] > arr[-2]:
        mx.append(arr_length - 1)
    elif arr[-1] < arr[-2]:
        mn.append(arr_length - 1)
    return np.array(mx), np.array(mn)


def read_files(path):
    files = glob.glob(path + '/*.csv')
    return files


def find_distance(infile):
    with open(infile, 'r') as i:
        head = i.readline().strip().split(',')[1:]
        data = np.genfromtxt(infile, delimiter=',', dtype=float)[1:]
        time, distance = [row[1] for row in data], [row[4] for row in data]
        x, y = [row[2] for row in data], [row[3] for row in data]
        i.close()

    time = np.array(time)
    # velocity = [0.0] * len(time)
    velocity = np.zeros(time.shape)
    # acceleration = [0.0] * len(time)
    acceleration = np.zeros(time.shape)

    velocity = calculate(velocity, distance, time)
    acceleration = calculate(acceleration, velocity, time)
    smoothed_velocity = savgol_filter(velocity, (lambda l: l - 1 if l % 2 == 0 else l)(len(velocity)), 8)
    smoothed_acceleration = savgol_filter(acceleration, (lambda l: l - 1 if l % 2 == 0 else l)(len(acceleration)), 8)

    gmx, gmn = find_local_maxima_and_minima(smoothed_velocity)
    # print(gmx)
    # print(gmn)

    plt.plot(time, smoothed_velocity, label="Graph X over Y")
    # plt.plot(time, velocity, label="Graph")

    if len(gmx) > 1:
        print(distance[gmx[1]])
        return distance[gmx[1]]
    else:
        print(distance[gmx[0]])
        return distance[gmx[0]]


def main():
    list_4m_files = read_files('test_4m')
    list_7m_files = read_files('test_7m')
    result_4m = []
    result_7m = []
    find_distance(file)
    #for file in list_4m_files:
     #   result_4m.append(find_distance(file))
    #for file in list_7m_files:
     #   result_7m.append(find_distance(file))
    #statistics = {'4m_min': np.min(result_4m), '4m_max': np.max(result_4m), '4m_mean': np.mean(result_4m),
     #             '4m_standard_deviation': np.std(result_4m), '7m_min': np.min(result_7m), '7m_max': np.max(result_7m),
      #            '7m_mean': np.mean(result_7m), '7m_standard_deviation': np.std(result_7m)}
    # print(result_4m)
    # print(result_7m)
    # print(statistics['4m_mean'])

    #with open('result.csv', 'w') as out:
     #   for key in statistics.keys():
      #      out.write("%s: %s\n" % (key, statistics[key]))
    #out.close()
    return result_4m, result_7m


if __name__ == '__main__':
    main()

plt.xlabel('time')
plt.ylabel('velocity')
plt.title('Graph: velocity over time')
plt.legend(["smoothed velocity", "velocity"])
plt.show()
