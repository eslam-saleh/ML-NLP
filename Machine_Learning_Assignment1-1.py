import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cost_function(xs, ys, thetas):
    z = np.power(((xs * thetas.T) - ys), 2)
    return np.sum(z) / len(xs)


def gradient_descent(xs, ys, thetas, a, iterations):
    temp_thetas = np.matrix(np.zeros(thetas.shape))
    iterations_cost = np.zeros(iterations)
    for i in range(iterations):
        error = (xs * thetas.T) - ys
        for j in range(thetas.shape[1]):
            term = np.multiply(error, xs[:, j])
            temp_thetas[0, j] = thetas[0, j] - ((a / len(xs)) * np.sum(term))
        thetas = temp_thetas
        iterations_cost[i] = cost_function(xs, ys, thetas)
    return thetas, iterations_cost

if __name__ == "__main__":
    url = "house_data.csv"
    data = pd.read_csv(url)
    while True:
        print('\n1- Linear Regression for 1 predictor  (sqft_living)')
        print('2- Linear Regression for 5 predictors (grade, bathrooms, lat, sqft_living, view)')
        print('0- Exit')

        print('\nYour Choice : ', end='')
        choice = str(input())
        if choice == '1':
            x = data.iloc[:, [5]]
            x_mean = x.mean()
            x_std = x.std()

            x = (x - x.mean()) / x.std()
            x.insert(0, 'ones', 1)

            y = data.iloc[:, [2]]
            y_mean = y.mean()
            y_std = y.std()

            y = (y - y.mean()) / y.std()

            x_matrix = np.matrix(x.values)
            y_matrix = np.matrix(y.values)
            c_matrix = np.matrix(np.zeros(2))

            tries = 2000
            while True:
                print('Enter alpha : ', end='')
                try:
                    alpha = float(input())
                except:
                    print('Try again\n')
                    continue

                g, cost = gradient_descent(x_matrix, y_matrix, c_matrix, alpha, tries)

                _, ax = plt.subplots()
                ax.plot(np.arange(tries), cost, 'r')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Cost')
                ax.set_title('alpha = ' + str(alpha))
                plt.show()

                print('\nPredict price (enter x to exit):')
                while True:
                    print('\nEnter sqft for living: ', end='')
                    try:
                        sqft_living = float((float(input()) - x_mean[0]) / x_std[0])
                        y_predicted = g[0, 0] + g[0, 1] * sqft_living
                        print('Predicted Price = ', float(y_predicted * y_std[0] + y_mean[0]))
                    except:
                        break
                x1 = np.linspace(x['sqft_living'].min(), x['sqft_living'].max())
                f = g[0, 0] + g[0, 1] * x1

                _, ax = plt.subplots()
                ax.scatter(x['sqft_living'], y['price'])
                ax.plot(x1, f, 'r')
                ax.set_xlabel('sqft_living')
                ax.set_ylabel('Price')
                plt.show()

                print('change alpha ? 1- yes 2- no')
                if input() != '1':
                    break

        elif choice == '2':
            x = data.iloc[:, [4, 5, 9, 11, 17]]
            x_mean = x.mean()
            x_std = x.std()

            x = (x - x.mean()) / x.std()
            x.insert(0, 'ones', 1)

            y = data.iloc[:, [2]]
            y_mean = y.mean()
            y_std = y.std()

            y = (y - y.mean()) / y.std()

            x_matrix = np.matrix(x.values)
            y_matrix = np.matrix(y.values)
            c_matrix = np.matrix(np.zeros(6))

            tries = 2000
            while True:
                print('Enter alpha : ', end='')
                try:
                    alpha = float(input())
                except:
                    print('Try again\n')
                    continue

                g, cost = gradient_descent(x_matrix, y_matrix, c_matrix, alpha, tries)

                _, ax = plt.subplots()
                ax.plot(np.arange(tries), cost, 'r')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Cost')
                ax.set_title('alpha = ' + str(alpha))
                plt.show()

                print('\nPredict price (enter x to exit):')
                while True:
                    try:
                        print('\nEnter # of bathrooms: ', end='')
                        bathrooms = float((int(input()) - x_mean[0]) / x_std[0])
                        print('Enter sqft for living: ', end='')
                        sqft_living = float((float(input()) - x_mean[1]) / x_std[1])
                        print('Enter # of views: ', end='')
                        view = float((int(input()) - x_mean[2]) / x_std[2])
                        print('Enter grade: ', end='')
                        grade = float((int(input()) - x_mean[3]) / x_std[3])
                        print('Enter lat: ', end='')
                        lat = float((float(input()) - x_mean[4]) / x_std[4])
                        y_predicted = g[0, 0] + g[0, 1] * bathrooms + g[0, 2] * sqft_living + g[0, 3] * view + \
                                      g[0, 4] * grade + g[0, 5] * lat
                        print('Predicted Price = ', float(y_predicted * y_std + y_mean))
                    except:
                        break

                x1 = np.linspace(x['bathrooms'].min(), x['bathrooms'].max())
                x2 = np.linspace(x['sqft_living'].min(), x['sqft_living'].max())
                x3 = np.linspace(x['view'].min(), x['view'].max())
                x4 = np.linspace(x['grade'].min(), x['grade'].max())
                x5 = np.linspace(x['lat'].min(), x['lat'].max())
                f = g[0, 0] + g[0, 1] * x1 + g[0, 2] * x2 + g[0, 3] * x3 + g[0, 4] * x4 + g[0, 5] * x5

                # _, axs = plt.subplots(2, 3)

                # axs[0, 0].scatter(x['bathrooms'], y['price'])
                # axs[0, 0].plot(x1, g[0, 0] + g[0, 1] * x1, 'r')
                # axs[0, 0].set_xlabel('Bathrooms')
                # axs[0, 0].set_ylabel('Price')
                #
                # axs[0, 1].scatter(x['sqft_living'], y['price'])
                # axs[0, 1].plot(x2, g[0, 0] + g[0, 2] * x2, 'r')
                # axs[0, 1].set_xlabel('sqft_living')
                # axs[0, 1].set_ylabel('Price')
                #
                # axs[0, 2].scatter(x['view'], y['price'])
                # axs[0, 2].plot(x3, g[0, 0] + g[0, 3] * x3, 'r')
                # axs[0, 2].set_xlabel('View')
                # axs[0, 2].set_ylabel('Price')
                #
                # axs[1, 0].scatter(x['grade'], y['price'])
                # axs[1, 0].plot(x4, g[0, 0] + g[0, 4] * x4, 'r')
                # axs[1, 0].set_xlabel('Grade')
                # axs[1, 0].set_ylabel('Price')
                #
                # axs[1, 1].scatter(x['lat'], y['price'])
                # axs[1, 1].plot(x5, g[0, 0] + g[0, 5] * x5, 'r')
                # axs[1, 1].set_xlabel('Lat')
                # axs[1, 1].set_ylabel('Price')

                _, ax1 = plt.subplots()
                ax1.scatter(x['bathrooms'], y['price'])
                ax1.plot(x1, g[0, 0] + g[0, 1] * x1, 'r')
                ax1.set_xlabel('Bathrooms')
                ax1.set_ylabel('Price')

                _, ax2 = plt.subplots()
                ax2.scatter(x['sqft_living'], y['price'])
                ax2.plot(x2, g[0, 0] + g[0, 2] * x2, 'r')
                ax2.set_xlabel('sqft_living')
                ax2.set_ylabel('Price')

                _, ax3 = plt.subplots()
                ax3.scatter(x['view'], y['price'])
                ax3.plot(x3, g[0, 0] + g[0, 3] * x3, 'r')
                ax3.set_xlabel('View')
                ax3.set_ylabel('Price')

                _, ax4 = plt.subplots()
                ax4.scatter(x['grade'], y['price'])
                ax4.plot(x4, g[0, 0] + g[0, 4] * x4, 'r')
                ax4.set_xlabel('Grade')
                ax4.set_ylabel('Price')

                _, ax5 = plt.subplots()
                ax5.scatter(x['lat'], y['price'])
                ax5.plot(x5, g[0, 0] + g[0, 5] * x5, 'r')
                ax5.set_xlabel('Lat')
                ax5.set_ylabel('Price')

                plt.show()

                print('change alpha ? 1- yes 2- no')
                if input() != '1':
                    break

        elif choice == '0':
            break
        else:
            print('Try again\n')
