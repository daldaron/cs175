import json
import cv2, random
import matplotlib.pyplot as plt
colors = [
		[100.,  100.,  100.],
		[100.,    0.,    0.],
		[150.,    0.,    0.],
		[200.,    0.,    0.],
		[255.,    0.,    0.],
		[100.,  100.,    0.],
		[150.,  150.,    0.],
		[200.,  200.,    0.],
		[255.,  255.,    0.],
		[  0.,  100.,   50.],
		[  0.,  150.,   75.],
		[  0.,  200.,  100.],
		[  0.,  255.,  125.],
		[  0.,   50.,  100.],
		[  0.,   75.,  150.],
		[  0.,  100.,  200.],
		[  0.,  125.,  255.],
		[100.,    0.,  100.],
		[150.,    0.,  150.],
		[200.,    0.,  200.],
		[255.,    0.,  255.]]

def draw_hand(canvas, joint, numclass =22, with_number = False, Edge = True):
	hand_map = [[0, 1],[1 , 2],[2 , 3],[3 , 4],
				[0, 5],[5 , 6],[6 , 7],[7 , 8],
				[0, 9],[9 ,10],[10,11],[11,12],
				[0,13],[13,14],[14,15],[15,16],
				[0,17],[17,18],[18,19],[19,20]]
	font = cv2.FONT_HERSHEY_SIMPLEX
	for i in range(len(joint)):
		if joint[i] == -1 :
			continue
		cv2.circle(canvas, tuple(joint[i][:2]), 2, colors[i], thickness=-1)
		if with_number:
			cv2.putText(canvas,str(i),tuple(joint[i][:2]),font,1,colors[i],thickness=.01)
	if Edge:
		for edge in hand_map:
			u,v = edge
			if joint[u] == -1 or joint[v] == -1:
				continue
			cv2.line(canvas,tuple(joint[u][:2]),tuple(joint[v][:2]),colors[v], 1)
	return canvas

def draw_hand_rescaled(canvas, joint, rescale=None, numclass=22, with_number=False, Edge=True):
	"""
	Added rescaling capabilities
	"""
	hand_map = [[0, 1],[1 , 2],[2 , 3],[3 , 4],
				[0, 5],[5 , 6],[6 , 7],[7 , 8],
				[0, 9],[9 ,10],[10,11],[11,12],
				[0,13],[13,14],[14,15],[15,16],
				[0,17],[17,18],[18,19],[19,20]]
	font = cv2.FONT_HERSHEY_SIMPLEX
	for i in range(len(joint)):
		if joint[i] == -1 :
			continue

		J_tuple = tuple([i//rescale for j in joint[i][:2]])


		cv2.circle(canvas, J_tuple, 4, colors[i], thickness=-1)
		if with_number:
			cv2.putText(canvas, str(i), J_tuple, font, 1, colors[i],1)
	if Edge:
		for edge in hand_map:
			u,v = edge
			if joint[u] == -1 or joint[v] == -1:
				continue

			J_tuple_u = tuple([i//rescale for j in joint[u][:2]])
			J_tuple_v = tuple([i//rescale for j in joint[v][:2]])

			cv2.line(canvas, J_tuple_u, J_tuple_v, colors[v],3)
	return canvas

def cmp(a, b):
	x, y = a.split("_"), b.split("_")
	if x[1] == y[1]:
		return int(x[2]) - int(y[2]);
	return int(x[1]) - int(y[1])

def main():
	Joints = json.load(open("Dataset/annotation.json","r"))
	names = Joints.keys()
	print names
	random.shuffle(names)
	LeftJoints = {}
	RightJoints = {}
	for name in Joints:
		# print name
		for pos in Joints[name]:
			if pos == -1:
				continue
			if int(pos[0]) == 10 and int(pos[1]) == 10:
				print name
	for name in Joints:
		if name[-1] == 'L':
			LeftJoints[name[:-2]] = Joints[name]
		if name[-1] == 'R':
			RightJoints[name[:-2]] = Joints[name]
	# LeftJointKey  = sorted(LeftJoints.keys() , cmp)
	# RightJointKey = sorted(RightJoints.keys(), cmp)
	for name in names:
		# if name[-1] == 'R':
		# 	continue
		# name = "032_2105_L"
		# print name
		# name = "AM_49827197_41_R"
		canvas = cv2.imread("Dataset/Images/" + name[:-2] + ".jpg")
		tmp = []
		print name
		for i in xrange(len(Joints[name])):
			tmp.append([int(Joints[name][i][0]), int(Joints[name][i][1])])
		# print tmp
		canvas = draw_hand(canvas, tmp)
		plt.imshow(canvas[:,:,::-1])
		plt.pause(1)


if __name__ == '__main__':
	main()
