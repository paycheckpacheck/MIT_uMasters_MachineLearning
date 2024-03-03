import cv2
import openpyxl
import numpy as np
import matplotlib.pyplot as plt

import time


class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, coords):
        ''' This function estimates the position of the object'''
        predicted_coords = []
        for coord in coords:
            measured = np.array([[np.float32(coord[0])], [np.float32(coord[1])]])
            self.kf.correct(measured)
            predicted = self.kf.predict()
            x, y = int(predicted[0]), int(predicted[1])
            predicted_coords.append((x, y))

        return predicted_coords

class Data_Getter:
    def __init__(self, filename, sheetname):
        # replace with the path to your .xlsx file
        self.filename = filename

        # open the workbook
        self.workbook = openpyxl.load_workbook(self.filename)

        # select the sheet containing the data
        self.sheet = self.workbook[str(sheetname)]  # replace 'Sheet1' with the name of your sheet

        # read the data from the specified columns
        self.data = []
        self.myrows= list(range(6, 250))

        self.get_data()



    def get_data(self):



        for i, row in enumerate(self.sheet.iter_rows(values_only=True)):

            if i in self.myrows:
                self.data.append(list(row))
                print("appended")

            else:print("not in myrows")



    def get_column(self, col):
        try:
            # get the specified column from the data
            if col < 1 or col > len(self.data[0]):
                print('Invalid column index')
            return [row[col-1] for row in self.data]
        except IndexError as IE:
            pass

    def get_row(self, row):

        try:
            # get the specified row from the data
            if row < 1 or row > len(self.data):
                print('Invalid row index')
            return self.data[row-1]
        except IndexError as IE:
            pass






#MAIN


#unload the data
stock_data_file = 'HistoricalData_1681579205818.xlsx'
Data = Data_Getter(stock_data_file, "HistoricalData_1681579205818")

#INIT KALMAN FILTER
Kalman = KalmanFilter()



stock_data = Data.data


time.sleep(3)
print(stock_data)


kalman_data, kalman_predictions = [], []











for data, predicitons in zip(kalman_data, kalman_predictions):
    # Plot the points
    plt.scatter(predicitons[0][0], predicitons[0][1], c='blue')
    plt.scatter(data[0][0], data[0][1], c='red')

    # Add axis labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

    # Show the plot
plt.show()











