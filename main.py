import cv2 as cv
import numpy as np
import os
import sys
import math
import fingerprint_enhancer

np.set_printoptions(threshold=sys.maxsize)

def segmentation(fingerprint, w):
	height, width = fingerprint.shape
	# block size
	global_threshold = np.var(fingerprint)*0.07
	

	greylevel_variance = np.zeros(fingerprint.shape)

	segmentation_boundary = np.ones(fingerprint.shape)


	for i in range(0, width, w):
		for j in range(0, height, w):
			block_topleft = [i,j]
			block_bottomright = [min(i+w,width), min(j+w, height)]
			greylevel_variance[block_topleft[1]:block_bottomright[1], block_topleft[0]:block_bottomright[0]] = np.var(fingerprint[block_topleft[1]:block_bottomright[1], block_topleft[0]:block_bottomright[0]])
	
	segmentation_boundary[greylevel_variance < global_threshold] = 0

	kernel = np.ones((2*w,2*w),np.uint8)
	# opening removes the noise outside the segmentation boundary line
	opening = cv.morphologyEx(segmentation_boundary, cv.MORPH_OPEN, kernel)
	# closing removes the noise inside the segmentation boundary line
	closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

	segmentation_boundary = closing

	segmented_image = fingerprint.copy()
	segmented_image[segmentation_boundary==0] = 255
	# segmented_image[segmentation_boundary==0] = 1
	return segmentation_boundary, segmented_image
	
def normalization(fingerprint, variance):
	# m0, v0 = 100.0, 100.0
	m0, v0 = 127.0, variance
	m,v = np.mean(fingerprint), np.var(fingerprint)
	# normalized_image = fingerprint.copy()
	normalized_image = np.zeros(fingerprint.shape)
	height, width = fingerprint.shape
	for i in range(0,width):
		for j in range(0,height):
			# calculate normalization for each pixel using normalization formula
			square_term = v0*((fingerprint[j][i]-m)**2)
			rootval = math.sqrt(square_term/v)
			if fingerprint[j][i] > m:
				normalized_image[j][i] = m0 + rootval
			else:
				normalized_image[j][i] = m0 - rootval
	return normalized_image

def orientation(fingerprint, block_size):
	Gx = cv.Sobel(fingerprint/255, cv.CV_64F, 0, 1, ksize=3)
	Gy = cv.Sobel(fingerprint/255, cv.CV_64F, 1, 0, ksize=3)

	# showimage("Gx", Gx)
	# showimage("Gy", Gy)

	height, width = fingerprint.shape

	V_x = np.zeros(fingerprint.shape)
	V_y = np.zeros(fingerprint.shape)
	for i in range(0, height):
		for j in range(0,width):
			si, ei = max(0, i-block_size//2), min(i+block_size//2, height)
			sj, ej = max(0, j-block_size//2), min(j+block_size//2, width)
			V_x[i, j] = np.sum(2*Gx[si: ei, sj: ej]*Gy[si: ei, sj: ej])
			V_y[i, j] = np.sum(Gx[si: ei, sj: ej]**2-Gy[si: ei, sj: ej]**2)

	Gaussian_std = 1
	Gaussian_Vx = cv.GaussianBlur(V_x, (2*block_size+1, 2*block_size+1), Gaussian_std)
	Gaussian_Vy = cv.GaussianBlur(V_y, (2*block_size+1, 2*block_size+1), Gaussian_std)
	# showimage("Gaussian_Vx", Gaussian_Vx)
	# showimage("Gaussian_Vy", Gaussian_Vy)

	orientation_angles = np.arctan2(Gaussian_Vx, Gaussian_Vy)/2
	for i in range(orientation_angles.shape[0]):
		for j in range(orientation_angles.shape[1]):
			orientation_angles[i][j] += np.pi/2

	return orientation_angles

def minutae_extraction(fingerprint, enhanced_mask, block_size):
	height, width = fingerprint.shape
	rows, cols = height-1, width-1
	# print(rows,cols)
	temp_image = fingerprint.copy()
	# print(type(fingerprint))
	
	for i in range(height):
		for j in range(width):
			if temp_image[i][j] == 0:
				temp_image[i][j] = 1
			else:
				temp_image[i][j] = 0
	


	minutae_image = cv.cvtColor(fingerprint.astype(np.uint8), cv.COLOR_GRAY2RGB)

	false_minutae_image = minutae_image.copy()

	pixelindex_to_cnp = {}
	# ms = 0
	for i in range(1,rows):
		for j in range(1,cols):
			pixel_pos = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
			if temp_image[i][j] == 1:
				pixel_vals=[]
				for i1,j1 in pixel_pos:
					pixel_vals.append(temp_image[i+j1][j+i1])
				cnp = 0
				for i1 in range(0,len(pixel_vals)-1):
					cnp += abs(pixel_vals[i1]-pixel_vals[i1+1])
				cnp = cnp//2
				# cnp = 0 implies isolated pixel
				# cnp = 1 implies ending
				# cmp = 2 implies no minutae
				# cmp = 3 implies bifurcation
				# cmp = 4 implies quadruple bifurcation or noise
				
				if cnp == 1:
					# print("here1")
					cv.circle(false_minutae_image, (j,i), radius=3, color=(0,0,255), thickness=1)
					pixelindex_to_cnp[(j,i)] = cnp
				elif cnp == 3:
					# print("here3")
					cv.circle(false_minutae_image, (j,i), radius=3, color=(0,255,0), thickness=1)
					pixelindex_to_cnp[(j,i)] = cnp
	showimage("minutae points with false positives", false_minutae_image)

	# print(len(pixelindex_to_cnp))

	true_minutaes = set()
	for k in pixelindex_to_cnp:
		# print(k)
		pos = [(0,1), (1,0), (0,0), (-1,0), (0,-1)]
		minutae_bool = True
		for i,j in pos:
			try:
				dir_x,dir_y = k[0] + i*block_size, k[1] + j*block_size
				if enhanced_mask[dir_y][dir_x] == 0:
					minutae_bool = False
					break
			except IndexError:
				minutae_bool = False
				break
		if minutae_bool:
			true_minutaes.add(k)
	# print(true_minutaes)

	# for index in true_minutaes:
	# 	if pixelindex_to_cnp[index] == 1:
	# 		cv.circle(minutae_image, index, radius=3, color=(0,0,255), thickness=1)
	# 	elif pixelindex_to_cnp[index] == 3:
	# 		cv.circle(minutae_image, index, radius=3, color=(0,255,0), thickness=1)
	
	def false_clustered_minutae_removal():
		minutae_coordinates = list(pixelindex_to_cnp)
		thresh_d = block_size/4
		got_minutae_cluster = False
		minutae_points_cluster = set()

		for i in range(1, len(minutae_coordinates)):
			for j in range(0, i):
				(x1, y1) = minutae_coordinates[i]
				(x2, y2) = minutae_coordinates[j]
				euclidean_distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
				if euclidean_distance <= thresh_d:
					got_minutae_cluster = True
					minutae_points_cluster.add((x1, y1))
					minutae_points_cluster.add((x2, y2))
		
		if not got_minutae_cluster:
			return False

		for _ in range(10):
			for i in range(len(minutae_coordinates)):
				if (x1, y1) not in minutae_points_cluster:
					for (x2, y2) in minutae_points_cluster:
						(x1, y1) = minutae_coordinates[i]
						euclidean_distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
						if euclidean_distance <= thresh_d:
							minutae_points_cluster.add((x1, y1))

		for (x1, y1) in minutae_points_cluster:
			del pixelindex_to_cnp[(x1,y1)]

		return True

	minuate_clusters_left = True
	while True:
		if not minuate_clusters_left:
			break
		minuate_clusters_left = false_clustered_minutae_removal()

	for index in true_minutaes:
		if index in pixelindex_to_cnp and pixelindex_to_cnp[index] == 1:
			cv.circle(minutae_image, index, radius=3, color=(0,0,255), thickness=1)
		elif index in pixelindex_to_cnp and pixelindex_to_cnp[index] == 3:
			cv.circle(minutae_image, index, radius=3, color=(0,255,0), thickness=1)

	return minutae_image, pixelindex_to_cnp

def minutae_image_points(fingerprint):
	block_size = 10
	segmented_boundary, segmented_image = segmentation(fingerprint, block_size)
	showimage("segmented_boundary", segmented_boundary)
	showimage("segmented_image", segmented_image)

	normalized_image = normalization(segmented_image, 100000)
	showimage("normalized_image", normalized_image)

	# orientation_angles = orientation(normalized_image, block_size)
	enhanced_image = fingerprint_enhancer.enhance_Fingerprint(normalized_image)
	showimage("enhanced_image", enhanced_image)

	thinned_image = cv.bitwise_not(cv.ximgproc.thinning(enhanced_image))
	showimage("thinned_image", thinned_image)

	# orientation_angles = orientation(thinned_image, block_size)
	
	new_block_size = 7
	enhanced_mask = segmentation(enhanced_image, new_block_size)[0]
	kernel = np.ones((2*new_block_size,2*new_block_size),np.uint8)
	# opening removes the noise outside the segmentation boundary line
	opening = cv.morphologyEx(enhanced_mask, cv.MORPH_OPEN, kernel)
	# closing removes the noise inside the segmentation boundary line
	enhanced_mask = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
	minutae_image, pixelindex_to_cnp = minutae_extraction(thinned_image.astype(np.float64), enhanced_mask, block_size)

	showimage("minutae points after false positives removal", minutae_image)

	def line_coordinates(i, j, linetan):
		if -1 <= linetan and linetan <= 1:
			p1, p2 = (int((-block_size/2) * linetan + j + block_size/2), i), (int((block_size/2) * linetan + j + block_size/2), i+block_size)
		else:
			p1, p2 = (j + block_size//2, int(i + block_size/2 + block_size/(2 * linetan))), (j - block_size//2, int(i + block_size/2 - block_size/(2 * linetan)))
		return (p1, p2)

	def draw_orientation_field(fingerprint, orientation_image, normalized_image, block_size, orientation_angles):
		for i in range(0, fingerprint.shape[0], block_size):
			for j in range(0, fingerprint.shape[1], block_size):
				si, ei = i, min(fingerprint.shape[0], i+block_size)
				sj, ej = j, min(fingerprint.shape[1], j+block_size)
				oriented_line = np.average(orientation_angles[si:ei, sj:ej])
				p1, p2 = line_coordinates(i, j, math.tan(oriented_line))
				cv.line(orientation_image, p1, p2, (0, 0, 255), 1)
		return orientation_image
	
	normalized_image = normalization(segmented_image, 5000.0)
	orientation_angles = orientation(normalized_image, block_size)

	orientation_image = cv.cvtColor((normalized_image).astype(np.uint8), cv.COLOR_GRAY2RGB)
	orientation_field = draw_orientation_field(normalized_image, orientation_image, normalized_image, block_size, orientation_angles)
	showimage("orientation_field", orientation_field)

	return minutae_image, pixelindex_to_cnp, orientation_angles

def alignment(template_image, query_image):
	_, template_xy, template_orientation = minutae_image_points(template_image)
	_, query_xy, query_orientation = minutae_image_points(query_image)
	
	block_size = 10

	acc = {}
	template_xy = list(template_xy)
	query_xy = list(query_xy)
	# print(template_xy)
	# print(query_xy)
	def new_round(num, multiple):
		return int(multiple * round(float(num)/multiple))

	for i in range(0,len(template_xy)):
		template_y, template_x = template_xy[i][0], template_xy[i][1]
		for j in range(0, len(query_xy)):
			query_y, query_x = query_xy[j][0], query_xy[j][1]
			del_theta = template_orientation[template_x][template_y] - query_orientation[query_x][query_y]
			delta_x = template_x - query_x*(np.cos(del_theta)) - query_y*(np.sin(del_theta))
			delta_y = template_y + query_x*(np.sin(del_theta)) - query_y*(np.cos(del_theta))
			del_theta, delta_x, delta_y = new_round(math.degrees(del_theta),5), new_round(delta_x, block_size), new_round(delta_y, block_size)
			if (del_theta, delta_x, delta_y) in acc:
				acc[(del_theta, delta_x, delta_y)] += 1
			else:
				acc[(del_theta, delta_x, delta_y)] = 1
	# for a in acc:
	#     print(a,acc[a])
	theta, x, y = max(acc, key=acc.get)
	# print("max: " + str(acc[(theta, x, y)]))
	align = math.radians(theta), x, y
	return align, (template_xy, template_orientation), (query_xy, query_orientation)

def minutae_pairing(minutae_set_t, minutae_set_q, align):
	block_size = 10
	minutae_pairs=[]
	flag_t = [0 for i in range(len(minutae_set_t[0]))]
	flag_q = [0 for i in range(len(minutae_set_q[0]))]
	pair_count = 0
	
	template_xy = minutae_set_t[0]
	query_xy = minutae_set_q[0]
	template_orientation = minutae_set_t[1]
	query_orientation = minutae_set_q[1]

	def euclidean_distance(p1, p2):
		x1,x2 = p1
		y1,y2 = p2
		return math.sqrt((x2-x1)**2 + (y2-y1)**2)
	
	atheta, ax, ay = align
	# atheta = abs(atheta)

	thresh_r = math.degrees(20)
	thresh_d = block_size

	for i in range(0,len(template_xy)):
		template_y, template_x = template_xy[i][0], template_xy[i][1]
		for j in range(0, len(query_xy)):
			query_y, query_x = query_xy[j][0], query_xy[j][1]
			
			del_theta = template_orientation[template_x][template_y] - query_orientation[query_x][query_y] + atheta
			delta_x = template_x - query_x*(np.cos(atheta)) - query_y*(np.sin(atheta)) + ax
			delta_y = template_y + query_x*(np.sin(atheta)) - query_y*(np.cos(atheta)) + ay

			# print(del_theta)

			if flag_t[i] == 0 and flag_q[j] == 0 and euclidean_distance((0, 0), (delta_x, delta_y)) <= thresh_d and del_theta <= thresh_r:
				flag_t[i] = 1
				flag_q[j] = 1
				pair_count += 1
				minutae_pairs.append((i,j))
			
	return pair_count, minutae_pairs

def load_db(db_directory_name):
	db_path = "FVC_2000/" + db_directory_name
	fp_imgs = os.listdir(db_path)
	# print(fp_imgs)
	personwise_fps = {}
	for img in fp_imgs:
		if int(img[0:3]) not in personwise_fps:
			personwise_fps[int(img[0:3])] = [db_path + "/" + img]
		else:
			personwise_fps[int(img[0:3])] += [db_path + "/" + img]

	for p in personwise_fps:
		personwise_fps[p] = sorted(personwise_fps[p])

	# print(personwise_fps)
	return personwise_fps

def fingerprint_matching(template_fingerprint_path, query_fingerprint_path):
	fingerprint1 = cv.imread(template_fingerprint_path, 0)
	showimage("original fingerprint1", fingerprint1)

	fingerprint2 = cv.imread(query_fingerprint_path, 0)
	showimage("original fingerprint2", fingerprint2)

	align, minutae_set_t, minutae_set_q = alignment(fingerprint1, fingerprint2)
	# deltheta, delx, dely = align
	# print(deltheta, delx, dely)
	match_count, minutae_pairs = minutae_pairing(minutae_set_t, minutae_set_q, align)
	template_minutae_count = len(minutae_set_t[0])
	query_minutae_count = len(minutae_set_q[0])

	return match_count, template_minutae_count, query_minutae_count


def showimage(imglabel, img):
	cv.imshow(imglabel, img)
	while True:
		k = cv.waitKey(0) & 0xFF
		if k == 27:         # wait for ESC key to exit
			cv.destroyAllWindows()
			break
	cv.destroyAllWindows()

if __name__ == "__main__":

	db = load_db("DB2_B")

	match_count, template_minutae_count, query_minutae_count = fingerprint_matching(db[101][0], db[101][1])

	print("minutae count in template fingerprint: " + str(template_minutae_count))	
	print("minutae count in query fingerprint: " + str(query_minutae_count))	
	print("count of total matched minutae: " + str(match_count))	


