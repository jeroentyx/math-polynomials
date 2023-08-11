import numpy as np
from numpy.polynomial import chebyshev
import math
from mpmath import chebyt, chop, taylor
from mpmath import *
from scipy import interpolate
import sympy as sym
import sys

"""
using the one he shown in the white board
x(t) = summation(Xi (t-tj)(ti-tj)
t is the value passed inside the interpolating function
xdata and tdata is used to create the interpolating function

"""


def summation(t, i, N, intervals):
    M = 1
    for j in range(0, N):
        if i == j:
            continue
        else:
            M *= ((t - intervals[j]) / (intervals[i] - intervals[j]))
    return M


def lagrange_e(t, data, intervals):
    N = len(data)
    L = 0
    for i in range(0, N):
        L += data[i] * summation(t, i, N, intervals)
    return L


def lagrange(t, xdata, tdata):
    no_of_pts = len(xdata)

    # if t = 0 (dividable by 0) error
    if t == 0:
        return 0

    # switch case
    if no_of_pts == 0:
        return 0
    elif no_of_pts == 1:
        return 0  # starting position is (0,0)
    elif no_of_pts == 2:
        basis0 = (t - tdata[1]) / (tdata[0] - tdata[1])
        basis1 = (t - tdata[0]) / (tdata[1] - tdata[0])
        return_value = (basis0 * xdata[0]) + (basis1 * xdata[1])
        return return_value
    elif no_of_pts == 3:
        basis0 = ((t - tdata[1]) * (t - tdata[2])) / ((tdata[0] - tdata[1]) * (tdata[0] - tdata[2]))
        basis1 = ((t - tdata[0]) * (t - tdata[2])) / ((tdata[1] - tdata[0]) * (tdata[1] - tdata[2]))
        basis2 = ((t - tdata[0]) * (t - tdata[1])) / ((tdata[2] - tdata[0]) * (tdata[2] - tdata[1]))
        return_value = (basis0 * xdata[0]) + (basis1 * xdata[1]) + (basis2 * xdata[2])
        return return_value
    elif no_of_pts == 4:
        basis0 = ((t - tdata[1]) * (t - tdata[2])
                  * (t - tdata[3])) / ((tdata[0] - tdata[1]) * (tdata[0] - tdata[2]) * (tdata[0] - tdata[3]))

        basis1 = ((t - tdata[0]) * (t - tdata[2])
                  * (t - tdata[3])) / ((tdata[1] - tdata[0]) * (tdata[1] - tdata[2]) * (tdata[1] - tdata[3]))

        basis2 = ((t - tdata[0]) * (t - tdata[1])
                  * (t - tdata[3])) / ((tdata[2] - tdata[0]) * (tdata[2] - tdata[1]) * (tdata[2] - tdata[3]))

        basis3 = ((t - tdata[0]) * (t - tdata[1])
                  * (t - tdata[2])) / ((tdata[3] - tdata[0]) * (tdata[3] - tdata[1]) * (tdata[3] - tdata[2]))
    else:
        print('invalid value')
        return -1
    # once reach here, return incorrect value
    return -1


'''
    natural cubic spline:
    take the param t , array of data(axis x or axis y data)
    and tdata, array of time data
'''

'''
    helper functions for natural cubic spline
    both taken from the numeric methods in python book
'''


def LUdecomp3(c, d, e):
    # n = size of d
    n = len(d)
    # range (start,stop,step)
    # generate numbers between 1 to n
    for k in range(1, n):
        lam = c[k - 1] / d[k - 1]
        d[k] = d[k] - lam * e[k - 1]
        c[k - 1] = lam
    return c, d, e


def LUsolve3(c, d, e, b):
    n = len(d)  # n is the size of d
    # k , start with 1, end with n
    for k in range(1, n):
        b[k] = b[k] - c[k - 1] * b[k - 1]

    b[n - 1] = b[n - 1] / d[n - 1]
    # k (start,stop,step)
    # start with n-2, stop at -1 , every step is -1
    for k in range(n - 2, -1, -1):
        b[k] = (b[k] - e[k] * b[k + 1]) / d[k]
    return b


"""
    Replace the xdata from the textbook with tdata
    so that the natural cubic spline interpolates with t instead
"""


def curvatures(tdata, ydata):
    # get the size of xData , minus by 1
    n = len(tdata) - 1
    c = np.zeros(n)  # array of zeros with size of n
    d = np.ones(n + 1)  # array of ones with size of n+1
    e = np.zeros(n)  # array of zeros with size of n
    k = np.zeros(n + 1)  # array of zeros with size of n+1
    # new_list = old_list[:] this duplicates
    # get 1,2,3, and so on , to n-1
    c[0:n - 1] = tdata[0:n - 1] - tdata[1:n]
    d[1:n] = 2.0 * (tdata[0:n - 1] - tdata[2:n + 1])  # what does this mean?
    e[1:n] = tdata[1:n] - tdata[2:n + 1]
    k[1:n] = 6.0 * (ydata[0:n - 1] - ydata[1:n]) / (tdata[0:n - 1] - tdata[1:n]) - 6.0 * (
            ydata[1:n] - ydata[2:n + 1]) / (tdata[1:n] - tdata[2:n + 1])
    LUdecomp3(c, d, e)
    LUsolve3(c, d, e, k)
    return k


def evalSpline(tdata, yData, k, x):
    def findSegment(tdata, x):
        iLeft = 0
        iRight = len(tdata) - 1
        while 1:
            if (iRight - iLeft) <= 1: return iLeft
            i = (iLeft + iRight) / 2
            if x < tdata[int(i)]:
                iRight = i
            else:
                iLeft = i

    i = findSegment(tdata, x)
    h = tdata[int(i)] - tdata[int(i + 1)]
    y = ((x - tdata[int(i + 1)]) ** 3 / h - (x - tdata[int(i + 1)]) * h) * k[int(i)] / 6.0 \
        - ((x - tdata[int(i)]) ** 3 / h - (x - tdata[int(i)]) * h) * k[int(i + 1)] / 6.0 \
        + (yData[int(i)] * (x - tdata[int(i + 1)])
           - yData[int(i + 1)] * (x - tdata[int(i)])) / h
    return y


"""
    Chebyshev polynomials
    measurement are taken at 1 + ti
    use Tn(t) = cos(n arc cos t)
    1 + ti will make it [0,2] , instead of the original [-1,1]
"""

if __name__ == "__main__":

    with open('partmove.txt', 'r') as file:
        no_of_points = int(file.readline())

        data_x = []
        data_y = []
        saved_t = []
        # no of points should be added by 1 for (pos 0,0)

        # create evenly interval between 0 and 2, with 8 data points(no of points)
        saved_t1 = np.linspace(0, 2, (no_of_points + 1))

        # print(saved_t1)

        # 0 ,1,2,3,4
        for i in range(1, (no_of_points + 1)):
            saved_t.append(float(saved_t1[i]))

        # generate sequence 0,1,2, if no of points is 3
        for i in range(0, no_of_points):
            point_xy = file.readline().split()
            data_x.append(float(point_xy[0]))
            data_y.append(float(point_xy[1]))

        final_line = file.readline().split()
        print("Q1: polynomial")
        # print("t: {}, predicted x: {}, y: {}".format(float(final_line[0]), float(final_line[1]), float(final_line[2])))

        x_val1 = lagrange_e(float(final_line[0]), data_x, saved_t)
        y_val2 = lagrange_e(float(final_line[0]), data_y, saved_t)
        epsilon = 0.000001
        x_discrepancy = (float(final_line[1]) - float(x_val1))
        y_discrepancy = (float(final_line[2]) - float(y_val2))

        if abs(x_discrepancy) < epsilon:
            x_discrepancy = 0
        if abs(x_discrepancy) < epsilon:
            y_discrepancy = 0

        # print("actual: {}, {}".format(float(x_val1), float(y_val2)))
        print("discrepancy:: {:f}, {:f}".format(float(x_discrepancy), float(y_discrepancy)))

# using Chebyshev polynomials as t parameters to a cubic spline interpolation

# do for natural cubic spline
with open('partmove.txt', 'r') as file:
    no_of_points = int(file.readline())

    data_x = []
    data_y = []
    saved_t = []
    # no of points should be added by 1 for (pos 0,0)

    # create evenly interval between 0 and 2, with 8 data points(no of points)
    saved_t1 = np.linspace(0, 2, (no_of_points + 1))

    # 0 ,1,2,3,4
    for i in range(1, (no_of_points + 1)):
        saved_t.append(float(saved_t1[i]))

    # generate sequence 0,1,2, if no of points is 3
    for i in range(0, no_of_points):
        point_xy = file.readline().split()
        data_x.append(float(point_xy[0]))
        data_y.append(float(point_xy[1]))

    final_line = file.readline().split()
    # print("t: {}, predicted x: {}, y: {}"
    #      .format(float(final_line[0]), float(final_line[1]), float(final_line[2])))
    expression_t = eval(final_line[0])

    # convert list to numpy array
    arr_xdata = np.array(data_x, float)
    arr_ydata = np.array(data_y, float)
    arr_tdata = np.array(saved_t, float)

    # equation will form two k coefficient for y and x-axis
    k_x = curvatures(arr_tdata, arr_xdata)
    k_y = curvatures(arr_tdata, arr_ydata)

    x_val1 = evalSpline(arr_tdata, data_x, k_x, float(final_line[0]))
    y_val2 = evalSpline(arr_tdata, data_y, k_y, float(final_line[0]))
    epsilon = 0.000001
    # y_val2 = lagrange(float(final_line[0]), saved_yData, saved_t)
    print("Q2: natural cubic spline")
    x_discrepancy = float(final_line[1]) - float(x_val1)
    y_discrepancy = float(final_line[2]) - float(y_val2)

    if abs(x_discrepancy) < epsilon:
        x_discrepancy = 0
    if abs(x_discrepancy) < epsilon:
        y_discrepancy = 0

    # print("actual: {}, {}".format(float(x_val1), float(y_val2)))
    print("discrepancy:: {:f}, {:f}".format(float(x_discrepancy), float(y_discrepancy)))

with open('partmove.txt', 'r') as file:
    no_of_points = int(file.readline())

    # generate the intervals with chebyshev
    coefs = [1, ] * (no_of_points + 1)
    coefs[-1] = 1
    C = np.polynomial.chebyshev.Chebyshev(coefs)
    R = C.roots()
    array_of_ones = np.ones(no_of_points)
    intervals = np.add(R, array_of_ones).tolist()

    saved_data_x = []
    saved_data_y = []
    saved_t = []

    # get the saved_x data and yData
    for i in range(0, no_of_points):
        point_xy = file.readline().split()
        saved_data_x.append(float(point_xy[0]))
        saved_data_y.append(float(point_xy[1]))

    final_line = file.readline().split()
    print("Q3: Chebyshev")
    # print("t: {}, predicted x: {}, y: {}"
    #       .format(float(final_line[0]), float(final_line[1]), float(final_line[2])))
    expression_t = eval(final_line[0])

    arr_xdata = np.array(saved_data_x, float)
    arr_ydata = np.array(saved_data_y, float)
    arr_tdata = np.array(intervals, float)

    # to do the chebyshev
    x_value1 = lagrange_e(float(final_line[0]), saved_data_x, intervals)
    y_value2 = lagrange_e(float(final_line[0]), saved_data_y, intervals)
    epsilon = 0.000001
    x_discrepancy = float(final_line[1]) - float(x_value1)
    y_discrepancy = float(final_line[2]) - float(y_value2)
    if abs(x_discrepancy) < epsilon:
        x_discrepancy = 0
    if abs(x_discrepancy) < epsilon:
        y_discrepancy = 0

    # print("actual: {}, {}".format(float(x_value1), float(y_value2)))
    print("discrepancy: {:f}, {:f}".format(float(x_discrepancy), float(y_discrepancy)))

# using this for legendre polynomial
# needs to modify this code to introduce legendre polynomial
with open('partmove.txt', 'r') as file:
    # using this for legendre polynomial
    no_of_points = int(file.readline())

    roots_container = (polyroots(taylor(lambda x: legendre(no_of_points, x), 0, no_of_points)[::-1]))

    floating_roots = []
    for i in range(0, no_of_points):
        floating_roots.append(float(roots_container[i]))

    for i in range(0, no_of_points):
        floating_roots[i] += float(1)

    saved_data_x = []
    saved_data_y = []
    saved_t = []

    # get the saved_x data and yData
    for i in range(0, no_of_points):
        point_xy = file.readline().split()
        saved_data_x.append(float(point_xy[0]))
        saved_data_y.append(float(point_xy[1]))

    final_line = file.readline().split()
    print("Q4: legendre polynomial")
    # print("t: {}, predicted x: {}, y: {}"
    #      .format(float(final_line[0]), float(final_line[1]), float(final_line[2])))
    expression_t = eval(final_line[0])

    x_value_legendre = lagrange_e(float(final_line[0]), saved_data_x, floating_roots)
    y_value_legendre = lagrange_e(float(final_line[0]), saved_data_y, floating_roots)
    epsilon = 0.000001
    x_discrepancy = float(final_line[1]) - float(x_value_legendre)
    y_discrepancy = float(final_line[2]) - float(y_value_legendre)

    if abs(x_discrepancy) < epsilon:
        x_discrepancy = 0
    if abs(x_discrepancy) < epsilon:
        y_discrepancy = 0

    # print("actual: {}, {}".format(float(x_value_legendre), float(y_value_legendre)))
    print("discrepancy:: {:f}, {:f}".format(float(x_discrepancy), float(y_discrepancy)))
# Question 5
with open('partmove.txt', 'r') as file:
    no_of_points = int(file.readline())
    saved_data_x = []
    saved_data_y = []
    saved_t = []

    saved_t1 = np.linspace(0, 2, (no_of_points + 1))
    # already have evenly space interval

    for i in range(1, (no_of_points + 1)):
        saved_t.append(float(saved_t1[i]))

    # generate sequence 0,1,2, if no of points is 3
    for i in range(0, no_of_points):
        point_xy = file.readline().split()
        saved_data_x.append(float(point_xy[0]))
        saved_data_y.append(float(point_xy[1]))

    final_line = file.readline().split()
    # print("Q5: Velocity:")
    # print("t: {}, predicted x: {}, y: {}".format(float(final_line[0]), float(final_line[1]), float(final_line[2])))

    x_val1 = lagrange_e(float(final_line[0]), saved_data_x, saved_t)
    y_val2 = lagrange_e(float(final_line[0]), saved_data_y, saved_t)
    epsilon = 0.000001
    x_discrepancy = (float(final_line[1]) - float(x_val1))
    y_discrepancy = (float(final_line[2]) - float(y_val2))

    if abs(x_discrepancy) < epsilon:
        x_discrepancy = 0
    if abs(x_discrepancy) < epsilon:
        y_discrepancy = 0

    # print("actual: {}, {}".format(float(x_val1), float(y_val2)))
    print("Q5: polynomial")
    print("discrepancy:: {:f}, {:f}".format(float(x_discrepancy), float(y_discrepancy)))
    userinput = input("press enter to exit")
