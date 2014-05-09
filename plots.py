import matplotlib
import matplotlib.pyplot as plt

def compute_error(y_true, y_pred, type='absolute'):
    if type=='absolute':
        return y_true - y_pred
    elif type=='relative':
        return (y_true - y_pred)/y_true

dates = matplotlib.dates.date2num(data1.index[test_idx])

plt.plot_date(dates, abs_error)
plt.xlabel('Date')
plt.ylabel('Absolute error (Million pounds of steam per hour)')
plt.title('Absolute error of steam demand prediction (SVR)')
plt.savefig('SVR_error.png')


