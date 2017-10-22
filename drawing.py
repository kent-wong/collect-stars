import tkinter as tk
import threading

def draw_bbox(canvas, bbox):
	return canvas.create_rectangle(*bbox)

def draw_red_ball(canvas, bbox):
	return canvas.create_oval(*bbox, fill='red')

def draw_gray_box(canvas, bbox):
	return canvas.create_rectangle(*bbox, fill='gray')

def draw_yellow_star(canvas, bbox):
	l, t, r, b = bbox
	x0 = l
	y0 = t + (b - t)*2/5
	x1 = r
	y1 = y0
	x2 = l + (r - l)/4
	y2 = b
	x3 = (l + r)/2
	y3 = t
	x4 = l + (r - l)*3/4
	y4 = b

	return canvas.create_polygon(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, fill='yellow')

def draw_pacman(canvas, bbox, angle=0, color='blue', outline='black', ratio=1):
	start = 35 + angle
	extent = 290
	canvas_id = canvas.create_arc(*bbox, start=start, extent=extent, fill=color, outline=outline)

	if ratio < 1:
		x0, y0, x1, y1 = bbox
		centerx = (x0 + x1) / 2
		centery = (y0 + y1) / 2
		canvas.scale(canvas_id, centerx, centery, ratio, ratio)
	
	return canvas_id

def draw_text(canvas, bbox, text, anchor):
	if text == None:
		return None

	x0, y0, x1, y1 = bbox
	if anchor == tk.CENTER:
		x = (x0 + x1) / 2
		y = (y0 + y1) / 2
	elif anchor == tk.N:
		x = (x0 + x1) / 2
		y = y0
	elif anchor == tk.S:
		x = (x0 + x1) / 2
		y = y1
	elif anchor == tk.W:
		x = x0
		y = (y0 + y1) / 2
		text = ' ' + text
	elif anchor == tk.E:
		x = x1
		y = (y0 + y1) / 2
		text = text + ' '
	else:
		assert False

	return canvas.create_text(x, y, text=text, font=('Times', 16), anchor=anchor)


draw_func_list = [draw_red_ball, draw_gray_box, draw_yellow_star]
drawing_function = {'red_ball': draw_red_ball,
			'yellow_star': draw_yellow_star,
			'gray_box': draw_gray_box,
			'pacman': draw_pacman}


"""A class that helps drawing shapes on a tkinter canvas"""
class DrawingManager(threading.Thread):
	def __init__(self, chessboard_size, square_size):
		threading.Thread.__init__(self)

		# drawing metrics
		self.n_rows, self.n_columns = chessboard_size
		self.square_width, self.square_height = square_size
		self.margin = 10

		# start running in new thread, use a lock to protect initializing stage
		self.lock = threading.Lock()
		self.lock.acquire()
		self.start()

	def bounding_box(self, index):
		"""Get a square bounding box coordinates"""

		row, column = index
		w, h = self.square_width, self.square_height

		left = w * column + self.margin
		top = h * row + self.margin
		right = left + w
		bottom = top + h
		return (left, top, right, bottom)

	# draw the whole 'chessboard' for this environment
	def draw_chessboard(self):
		for row in range(self.n_rows):
			for column in range(self.n_columns):
				bbox = self.bounding_box((row, column))
				draw_bbox(self.canvas, bbox)
	
	@staticmethod
	def _tag_of_square(index):
		return 'square@' + str(index)
	
	@staticmethod
	def _tag_of_name(name):
		return name + '@all'

	@staticmethod
	def _tag_of_trace_square(index):
		return 'trace@' + str(index)

	@staticmethod
	def _tag_of_trace():
		return 'trace@all'

	@staticmethod
	def _tag_of_text_square(index):
		return 'text_square@' + str(index)

	@staticmethod
	def _tag_of_text():
		return 'text@all'

	@staticmethod
	def _tag_of_agent():
		return 'agent'

	def draw_object(self, name, index):
		bbox = self.bounding_box(index)

		tag = self._tag_of_square(index)
		all_tag = self._tag_of_name(name)

		self.canvas.delete(tag)
		canvas_id = drawing_function[name](self.canvas, bbox)

		self.canvas.addtag_withtag(tag, canvas_id)
		self.canvas.addtag_withtag(all_tag, tag)

		return canvas_id

	def delete_object(self, name, index):
		tag = self._tag_of_square(index)
		#self.canvas.dtag(tag, tag_of_cell)
		self.canvas.delete(tag)

	#def is_object_on_cell(self, obj_type, index_or_id):
	#	cell_id = self.grid.insure_id(index_or_id)
	#	tag = obj_type + '_' + str(cell_id)
	#	
	#	return len(self.canvas.find_withtag(tag))

	#def delete_objects_on_cell(self, index_or_id):
	#	cell_id = self.grid.insure_id(index_or_id)
	#	tag_of_cell = str(cell_id) + '_cell_id'
	#	self.canvas.delete(tag_of_cell)

	def draw_text(self, index, text_dict):
		bbox = self.bounding_box(index)
		tag = self._tag_of_text_square(index)
		tag_all = self._tag_of_text()

		self.canvas.delete(tag)
		anchor_dict = {"N":tk.N, "S":tk.S, "W":tk.W, "E":tk.E, "C":tk.CENTER}
		for anchor, text in text_dict.items():
			canvas_id = draw_text(self.canvas, bbox, text, anchor_dict[anchor])
			self.canvas.addtag_withtag(tag, canvas_id)
			self.canvas.addtag_withtag(tag_all, tag)
		
	def delete_text(self, index):
		tag = self._tag_of_text_square(index)
		self.canvas.delete(tag)
		
	def delete_all_text(self):
		tag_all = self._tag_of_text()
		self.canvas.delete(tag_all)

	#def draw_text_list(self, index_or_id, text_anchor_list):
	#	bbox = self.bounding_box(index_or_id)
	#	cell_id = self.grid.insure_id(index_or_id)
	#	tag = 'text_list' + str(cell_id)

	#	self.canvas.delete(tag)
	#	for text, anchor in text_anchor_list:
	#		canvas_id = draw_text(self.canvas, bbox, text, anchor)
	#		self.canvas.addtag_withtag(tag, canvas_id)

	def draw_trace(self, index, facing, ratio=0.5):
		angle = self._angle_from_facing(facing)
		bbox = self.bounding_box(index)
		color = '#f8fff8'

		tag = self._tag_of_trace_square(index)
		all_tag = self._tag_of_trace()

		# delete old trace on this square
		self.canvas.delete(tag)

		trace_canvas_id = draw_pacman(self.canvas, bbox, angle, color, color, ratio)
		# move the 'trace' to the lowest layer of canvas elements, so it doesn't block other elements(such as text)
		self.canvas.tag_lower(trace_canvas_id, None)

		self.canvas.addtag_withtag(tag, trace_canvas_id)
		self.canvas.addtag_withtag(all_tag, trace_canvas_id)

	def delete_trace(self):
		all_tag = self._tag_of_trace()
		self.canvas.delete(all_tag)
		
	@staticmethod
	def _angle_from_facing(facing):
		angle = {'N':90, 'S':270, 'W':180, 'E':0}
		return angle[facing]

	def draw_agent(self, index, facing):
		angle = self._angle_from_facing(facing)
		bbox = self.bounding_box(index)
		tag = self._tag_of_agent()
		canvas_id = draw_pacman(self.canvas, bbox, angle)
		self.canvas.addtag_withtag(tag, canvas_id)
		
	def remove_agent(self):
		tag = self._tag_of_agent()
		self.canvas.delete(tag)

	def rotate_agent(self, index, facing):
		# tkinter doesn't support rotate a canvas element, so delete then draw again
		self.remove_agent()
		self.draw_agent(index, facing)

	def move_agent(self, index_src, index_dest):
		bbox_src = self.bounding_box(index_src)
		bbox_dest = self.bounding_box(index_dest)

		move_x = bbox_dest[0] - bbox_src[0]
		move_y = bbox_dest[1] - bbox_src[1]

		tag = self._tag_of_agent()
		self.canvas.move(tag, move_x, move_y)

	def run(self):
		# create tkinter window
		window = tk.Tk()
		window.title('Reinforcement Learning Grid Environment')

		# set window size
		width = self.n_columns * self.square_width + self.margin
		height = self.n_rows * self.square_height + self.margin
		window.geometry(str(width)+'x'+str(height)+'+500+200')

		# set canvas
		canvas = tk.Canvas(window, width=width, height=height)
		canvas.grid(row=0, column=0)

		# save parameters
		self.window = window
		self.canvas = canvas

		self.lock.release()

		# enter window mainloop
		self.window.mainloop()

	def wait(self):
		# wait untill tkinter window is established
		self.lock.acquire()
		self.lock.release()

		return self

