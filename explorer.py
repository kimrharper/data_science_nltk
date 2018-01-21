#	written by Matt Borthwick for "Data Exploration in Python" presentation, Jan 17, 2018

from _csv import reader
import matplotlib.pyplot as plt
import numpy as np

#	exapmle usage:
#
#		from explorer import FakeDataExplorer
#		ex = FakeDataExplorer("fake-data.csv")
#		print(ex.keys())
#		ex.histograms("revenue")
#		ex.scatter_plots("growth", "score2")

class FakeDataExplorer(dict):

	def __init__(self, data_filename):
		""" load csv data into a dict of lists """		
		with open(data_filename) as f:
			rows = reader(f)
			#	headers will be the dict keys
			headers = next(rows)[1:] #	omit the first column, which is names
			#	the dict values will be lists of the column values
			super().__init__(zip(headers, zip(*(self.process_row(row, headers) for row in rows))))

	def process_row(self, row, headers):
		#	missing values will be replaced by zero, omitting the first column
		#
		#	if the columns hold different types of data, this could instead
		#	call different functions to process the cells in each column
		return (self.process_cell(cell) for cell in row[1:])

	@staticmethod			
	def process_cell(cell):
		#	fill missing cells with zeroes
		return float(cell) if cell else 0
			
	combinations = ((False, False), (True, False), (False, True), (True, True))

	def histograms(self, key, n_bins=200):
		""" plot four separate histograms of self[key], showing each combination of linear and logarithmic axes
			and compare each histogram to a normal distribution
		"""
		
		normal_curve = self.calc_normal_curve(self[key], n_bins)

		plt.figure(0)
		plt.clf() #	clear figure
		for index, use_log_scale in enumerate(self.combinations, 1):
			plt.subplot(2, 2, index).tick_params(labelsize="x-small")
			self.single_histogram(key, use_log_scale=use_log_scale, normal_curve=(None if use_log_scale[0] else normal_curve), n_bins=n_bins)
		plt.suptitle(key)
		plt.show(block=False)
		
	label_combinations = ((False, True), (False, False), (True, True), (False, False))

	def scatter_plots(self, key_x, key_y):
		"""	plot four separate scatterplots of key_y vs. key_x, showing each combination of linear and logarithmic axes
		"""
		
		plt.figure(1)
		plt.clf()
		for index, use_log_scale in enumerate(self.combinations, 1):
			plt.subplot(2, 2, index).tick_params(labelsize="x-small")
			self.single_scatter_plot(key_x, key_y, use_log_scale=use_log_scale, show_labels=(index > 2, index % 2))
		plt.show(block=False)

	def single_histogram(self, key, use_log_scale=(False, False), normal_curve=None, n_bins=200):
		"""	plot a single histogram of self[key], with log or linear axes as specified,
			along with a curve showing a normal distribution with the same mean and stdev as self[key] 
		"""
		vals = self[key]
							
		if use_log_scale[0]: #	logarithmic x-axis
			plt.xscale("log")
			#	show only positive values
			vals = tuple(val for val in vals if val > 0)
			#	make bins equally sized on a logarithmic axis
			bins = np.geomspace(min(vals), max(vals), n_bins)
		else: #	linear x-axis, just use equally spaced bins
			bins = n_bins
			
		plt.hist(vals, bins=bins, log=use_log_scale[1])

		if normal_curve:
			plt.plot(*normal_curve, color=(1,0,0,0.3), linewidth=2)

	def calc_normal_curve(self, vals, n_bins):
		"""	calculate a normal distribution with the same mean and stdev as vals """
		stdev = np.std(vals)
		z = np.linspace(-4, 4, 200)
		x = z * stdev + np.mean(vals)
		y = len(vals) / n_bins * np.pi * np.exp(-0.5 * z * z)
		return x, y
		
	def single_scatter_plot(self, key_x, key_y, use_log_scale=(False, False), size=0.1, show_labels=(True, True), **keywords):
		""" plot self[key_y] vs. self[key_x] and calculate the Pearson correlation coefficient between them """
		x_vals = self[key_x]
		y_vals = self[key_y]
		
		#	show only positive values when using logarithmic axes
		if use_log_scale[0]:
			plt.xscale("log")
			x_vals, y_vals = zip(*((x, y) for x, y in zip(x_vals, y_vals) if x > 0))
		if use_log_scale[1]:
			plt.yscale("log")
			x_vals, y_vals = zip(*((x, y) for x, y in zip(x_vals, y_vals) if y > 0))
			corr_y_vals = np.log(y_vals)

		vals_for_corr = (np.log(vals) if use_log else vals for vals, use_log in zip((x_vals, y_vals), use_log_scale))
		r_squared = "$r^2$ = {:.5}".format(np.corrcoef(*vals_for_corr)[0,1] ** 2)

		plt.scatter(x_vals, y_vals, s=size, c=(0,0,0.6,0.3), label=r_squared, **keywords)
		plt.legend(markerscale=0, markerfirst=False, frameon=False)

		for show_label, labeller, key in zip(show_labels, (plt.xlabel, plt.ylabel), (key_x, key_y)):
			if show_label:
				labeller(key)
