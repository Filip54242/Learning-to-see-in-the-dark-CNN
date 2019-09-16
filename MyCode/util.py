import csv
from statistics import mean

import matplotlib.pyplot as plt


def get_from_csv(file_path):
    reader = csv.reader(open(file_path), delimiter=' ')
    return [[float(element) for element in elements[0].split(",")] for elements in reader][0]


def get_average_from_csv(file_path):
    return mean(get_from_csv(file_path))


def draw_plots(file_paths, plot_title="NEW PLOT", names=None):
    if names is None:
        names = ["line %d" % index for index in range(len(file_paths))]
    plt.title(plot_title)
    for file_path in file_paths:
        y = get_from_csv(file_path)
        x = [index for index in range(len(y))]
        plt.plot(x, y, label=names[file_paths.index(file_path)])
    plt.legend()
    plt.show()

#
# draw_plots(("/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/psnr_baseline.csv",
#             "/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/psnr_with_filter.csv",
#             "/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/psnr__noise.csv"), "PSNR",
#            ["Baseline", "Filtered", "Learned noise"])
#
# draw_plots(("/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/ssim_baseline.csv",
#             "/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/ssim_with_filter.csv",
#             "/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/ssim_noise.csv"), "SSIM",
#            ["Baseline", "Filtered", "Learned noise"])

draw_plots(("/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/psnr__noise.csv",), "PSNR",
           ["Learned noise"])
draw_plots(("/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/ssim_noise.csv",), "SSIM",
           ["Learned noise"])


# print("PSNR baseline:",
#       get_average_from_csv("/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/psnr_baseline.csv"))
# print("PSNR with filter:",
#       get_average_from_csv("/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/psnr_with_filter.csv"))
# print("PSNR with noise:",
#       get_average_from_csv("/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/psnr__noise.csv"))
# print("SSIM baseline:",
#       get_average_from_csv("/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/ssim_baseline.csv"))
# print("SSIM with filter:",
#       get_average_from_csv("/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/ssim_with_filter.csv"))
# print("SSIM with noise:",
#       get_average_from_csv("/home/student/Documents/Learning-to-see-in-the-dark-CNN/MyCode/ssim_noise.csv"))
