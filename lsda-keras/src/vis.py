import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_detections(img, boxes, ids):
	# Create the plot, add the image
	fig, ax = plt.subplots(1, figsize=(15,15))
	ax.imshow(img)

	# Add the rectangles to the plot
	for box, id_ in reversed(list(zip(boxes, ids))):
		rect = patches.Rectangle(
			(box[0][0], box[0][1]),
			box[0][2] - box[0][0],
			box[0][3] - box[0][1],
			linewidth=1,
			edgecolor="r" if id_ > 200 else "b",
			facecolor="none"
		)
		ax.add_patch(rect)

		# Add textbox to the plot
		y_txt = min(img.shape[1] - 15, box[0][3] + 5)
		ax.annotate(
			"{}: {:.4f}".format(id_, box[1]),
			(box[0][0], y_txt),
			transform=ax.transAxes,
			fontsize=12,
			verticalalignment='top',
			bbox={
				"facecolor": (0.7, 0.9, 0.7),
				"alpha": 0.5
			}
		)
	plt.show()
	